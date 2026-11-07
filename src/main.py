import os

if os.path.exists("/dev/accel0") or os.environ.get("TPU_NAME") or os.environ.get("COLAB_TPU_ADDR"):
    os.environ["JAX_PLATFORMS"] = "tpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
    os.environ["XLA_FLAGS"] = "--xla_tpu_spmd_rng_bit_generator_unsafe=true --xla_tpu_enable_data_parallel_all_reduce_opt=true"
else:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import yaml
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed
from tensorboardX import SummaryWriter
from pycocotools import mask as maskUtils

cv2.setNumThreads(0)
jax.config.update("jax_default_matmul_precision", "bfloat16")

devices = []
try:
    devices = jax.devices()
    is_accel = devices[0].platform in ("tpu", "gpu")
except Exception:
    is_accel = False

print(f"{'Accel' if is_accel else 'CPU'} | {len(devices)} devices")

IS_KAGGLE = os.path.exists("/kaggle/input")
CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

if IS_KAGGLE:
    BASE = cfg['paths']['kaggle_base']
    CHUNK_DIR = cfg['paths']['kaggle_chunks']
    VAL_DIR = cfg['paths']['kaggle_chunks_val']
else:
    BASE = cfg['paths']['local_base']
    CHUNK_DIR = cfg['paths']['local_chunks']
    VAL_DIR = cfg['paths']['local_chunks_val']

HP = cfg['hyperparameters']
IMG_SIZE = HP['img_size']
BASE_CH = HP['base_channels']
PER_DEVICE_BATCH = HP['per_device_batch']
EPOCHS = HP['epochs']
LR = HP['learning_rate']
SEED = HP['seed']
EMA_DECAY = HP['ema_decay']
WARMUP = HP['warmup_steps']
MIN_AREA = int(HP['min_area'])
POS_THR = float(HP['pos_thr'])
WS_H = float(HP['watershed_h'])
SPLIT_AREA = int(HP['split_area'])
W_FOCAL = float(HP['w_focal'])
W_DICE = float(HP['w_dice'])
W_BND = float(HP['w_boundary'])

NUM_DEVICES = max(1, jax.local_device_count())
GLOBAL_BATCH_SIZE = PER_DEVICE_BATCH * NUM_DEVICES

CHECKPOINT_DIR = "/kaggle/working/checkpoints" if IS_KAGGLE else "./checkpoints"
LOG_DIR = "/kaggle/working/logs" if IS_KAGGLE else "./tensorboard"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

assert os.path.isdir(CHUNK_DIR), f"Missing {CHUNK_DIR}."
assert os.path.isdir(VAL_DIR), f"Missing {VAL_DIR}."

class SolarFilamentDataset(Dataset):
    def __init__(self, chunk_dir):
        self.chunk_dir = chunk_dir
        self.files = sorted([f for f in os.listdir(chunk_dir) if f.startswith('img_') and f.endswith('.npy')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = self.files[idx]
        idx_str = img_file.replace('img_', '').replace('.npy', '')
        mask_file = f"mask_{idx_str}.npy"
        img = np.load(os.path.join(self.chunk_dir, img_file))
        mask = np.load(os.path.join(self.chunk_dir, mask_file))
        return img, mask

train_dataset = SolarFilamentDataset(CHUNK_DIR)
val_dataset = SolarFilamentDataset(VAL_DIR)

train_loader = DataLoader(
    train_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=True, drop_last=True,
    num_workers=0, pin_memory=False,
)
val_loader = DataLoader(
    val_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=False, drop_last=False,
    num_workers=0, pin_memory=False,
)

def augment_jax(img, mask, key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    flip_lr = jax.random.bernoulli(k1)
    img = jnp.where(flip_lr, jnp.flip(img, axis=1), img)
    mask = jnp.where(flip_lr, jnp.flip(mask, axis=1), mask)
    flip_ud = jax.random.bernoulli(k2)
    img = jnp.where(flip_ud, jnp.flip(img, axis=0), img)
    mask = jnp.where(flip_ud, jnp.flip(mask, axis=0), mask)
    k_rot = jax.random.randint(k3, (), 0, 4)
    img = jnp.where(k_rot == 0, img, jnp.where(k_rot == 1, jnp.rot90(img, 1), jnp.where(k_rot == 2, jnp.rot90(img, 2), jnp.rot90(img, 3))))
    mask = jnp.where(k_rot == 0, mask, jnp.where(k_rot == 1, jnp.rot90(mask, 1), jnp.where(k_rot == 2, jnp.rot90(mask, 2), jnp.rot90(mask, 3))))
    gamma = jax.random.uniform(k4, (), minval=0.8, maxval=1.2)
    img = jnp.clip(img ** gamma, 0.0, 1.0)
    return img, mask

@jax.jit
def preprocess_train(x, y, keys):
    x = x.astype(jnp.float32) / 255.0
    y = y.astype(jnp.float32)
    x, y = jax.vmap(jax.vmap(augment_jax))(x, y, keys)
    return jnp.expand_dims(x, 2).astype(jnp.bfloat16), jnp.expand_dims(y, 2).astype(jnp.bfloat16)

@jax.jit
def preprocess_val(x, y):
    x = x.astype(jnp.float32) / 255.0
    y = y.astype(jnp.float32)
    return jnp.expand_dims(x, 2).astype(jnp.bfloat16), jnp.expand_dims(y, 2).astype(jnp.bfloat16)

class ConvBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm

    def __init__(self, in_ch, out_ch, key):
        k1, k2 = jax.random.split(key, 2)
        self.conv1 = eqx.nn.Conv2d(in_ch, out_ch, 3, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv2d(out_ch, out_ch, 3, padding=1, key=k2)
        self.norm1 = eqx.nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = eqx.nn.GroupNorm(min(8, out_ch), out_ch)

    def __call__(self, x):
        x = jax.nn.relu(self.norm1(self.conv1(x)))
        x = jax.nn.relu(self.norm2(self.conv2(x)))
        return x

class UNet(eqx.Module):
    down1: ConvBlock
    down2: ConvBlock
    down3: ConvBlock
    down4: ConvBlock
    pool: eqx.nn.MaxPool2d
    up_conv4: eqx.nn.ConvTranspose2d
    up_conv3: eqx.nn.ConvTranspose2d
    up_conv2: eqx.nn.ConvTranspose2d
    up_conv1: eqx.nn.ConvTranspose2d
    up1: ConvBlock
    up2: ConvBlock
    up3: ConvBlock
    up4: ConvBlock
    out_conv: eqx.nn.Conv2d

    def __init__(self, in_channels=1, out_channels=1, base=64, key=jax.random.PRNGKey(0)):
        keys = jax.random.split(key, 15)
        c1, c2, c3, c4 = base, base*2, base*4, base*8
        self.pool = eqx.nn.MaxPool2d(2, 2)
        self.down1 = ConvBlock(in_channels, c1, keys[0])
        self.down2 = ConvBlock(c1, c2, keys[1])
        self.down3 = ConvBlock(c2, c3, keys[2])
        self.down4 = ConvBlock(c3, c4, keys[3])

        self.up_conv4 = eqx.nn.ConvTranspose2d(c4, c3, 2, 2, key=keys[4])
        self.up4 = ConvBlock(c4 + c3, c3, keys[5])
        self.up_conv3 = eqx.nn.ConvTranspose2d(c3, c2, 2, 2, key=keys[6])
        self.up3 = ConvBlock(c3 + c2, c2, keys[7])
        self.up_conv2 = eqx.nn.ConvTranspose2d(c2, c1, 2, 2, key=keys[8])
        self.up2 = ConvBlock(c2 + c1, c1, keys[9])
        self.up_conv1 = eqx.nn.ConvTranspose2d(c1, c1//2, 2, 2, key=keys[10])
        self.up1 = ConvBlock(c1//2 + c1, c1//2, keys[11])
        self.out_conv = eqx.nn.Conv2d(c1//2, out_channels, 1, key=keys[12])

    def __call__(self, x):
        x1 = self.down1(x); p1 = self.pool(x1)
        x2 = self.down2(p1); p2 = self.pool(x2)
        x3 = self.down3(p2); p3 = self.pool(x3)
        x4 = self.down4(p3); p4 = self.pool(x4)

        u4 = self.up_conv4(p4)
        x4u = self.up4(jnp.concatenate([u4, x4], axis=0))
        u3 = self.up_conv3(x4u)
        x3u = self.up3(jnp.concatenate([u3, x3], axis=0))
        u2 = self.up_conv2(x3u)
        x2u = self.up2(jnp.concatenate([u2, x2], axis=0))
        u1 = self.up_conv1(x2u)
        x1u = self.up1(jnp.concatenate([u1, x1], axis=0))
        return self.out_conv(x1u)

def focal_loss(preds, targets, alpha=0.8, gamma=2.0):
    bce = optax.sigmoid_binary_cross_entropy(preds, targets)
    pt = jnp.exp(-bce)
    return jnp.mean(alpha * (1 - pt) ** gamma * bce)

def dice_loss(preds, targets):
    preds = jax.nn.sigmoid(preds).reshape(-1)
    targets = targets.reshape(-1)
    inter = jnp.sum(preds * targets)
    return 1.0 - (2.0 * inter + 1e-5) / (jnp.sum(preds) + jnp.sum(targets) + 1e-5)

def boundary_loss(preds, targets):
    te = jnp.abs(jnp.roll(targets, 1, axis=2) - targets) + jnp.abs(jnp.roll(targets, 1, axis=3) - targets)
    te = jnp.clip(te, 0, 1)
    bce = optax.sigmoid_binary_cross_entropy(preds, targets)
    return jnp.mean(bce * (1.0 + 5.0 * te))

def make_train_step(optimizer, ema_decay):
    def loss_fn(model, x, y):
        model_bf16 = jax.tree_util.tree_map(lambda p: p.astype(jnp.bfloat16) if eqx.is_array(p) else p, model)
        preds = jax.vmap(model_bf16)(x)
        return W_FOCAL * focal_loss(preds, y) + W_DICE * dice_loss(preds, y) + W_BND * boundary_loss(preds, y)

    @eqx.filter_pmap(axis_name='batch')
    def train_step(model, ema_model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        grads = jax.tree_util.tree_map(lambda g: jax.lax.pmean(g, axis_name='batch'), grads)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        ema_model = jax.tree_util.tree_map(lambda e, m: ema_decay * e + (1.0 - ema_decay) * m, ema_model, model)
        return model, ema_model, opt_state, jax.lax.pmean(loss, axis_name='batch')

    return train_step

@eqx.filter_jit
def eval_step_bf16(model_bf16, x, y):
    return dice_loss(jax.vmap(model_bf16)(x), y)

def unbroadcast(pmodel):
    def fn(x):
        if eqx.is_array(x) and x.ndim > 0 and x.shape[0] == NUM_DEVICES:
            return x[0]
        return x
    return jax.tree_util.tree_map(fn, pmodel)

def validate(model_single):
    model_bf16 = jax.tree_util.tree_map(lambda p: p.astype(jnp.bfloat16) if eqx.is_array(p) else p, model_single)
    losses = []
    for imgs, masks in val_loader:
        x_in = jnp.array(imgs.numpy())
        y_in = jnp.array(masks.numpy())
        x, y = preprocess_val(x_in, y_in)
        losses.append(float(eval_step_bf16(model_bf16, x, y)))
    return float(np.mean(losses))

def watershed_labels(binary_mask):
    dist = ndimage.distance_transform_edt(binary_mask)
    local_max = h_maxima(dist, h=WS_H)
    markers = ndimage.label(local_max)[0].astype(np.int32)
    cc, _ = ndimage.label(binary_mask)
    markers = np.where((markers == 0) & (binary_mask > 0), cc + markers.max() + 1, markers)
    return watershed(-dist, markers, mask=binary_mask).astype(np.int32)

def generate_submission(model):
    print("Generating submission...")
    test_dir = os.path.join(BASE, cfg['paths']['test_img'])
    exts = ('.jpeg', '.jpg', '.png', '.tif', '.tiff')
    test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(exts)])
    results = []

    model_bf16 = jax.tree_util.tree_map(lambda p: p.astype(jnp.bfloat16) if eqx.is_array(p) else p, model)

    @eqx.filter_jit
    def predict_batch(m, imgs):
        return jax.nn.sigmoid(jax.vmap(m)(imgs))

    hann = np.hanning(IMG_SIZE)
    ww = np.maximum(np.outer(hann, hann), 0.01)
    stride = 256
    window = IMG_SIZE

    def tta_predict(batch, N):
        rots = [jnp.rot90(batch, i, axes=(2, 3)) for i in range(4)]
        rots_flip = [jnp.flip(r, axis=3) for r in rots]
        all_views = jnp.concatenate(rots + rots_flip, axis=0)

        preds = predict_batch(model_bf16, all_views)

        p_rots = preds[:4*N]
        p_flips = preds[4*N:]

        acc = np.zeros((N, 1, window, window), dtype=np.float32)
        for i in range(4):
            p = p_rots[i*N:(i+1)*N]
            p = jnp.rot90(p, -i, axes=(2, 3))
            acc += np.array(p)

            p_f = p_flips[i*N:(i+1)*N]
            p_f = jnp.rot90(p_f, -i, axes=(2, 3))
            p_f = jnp.flip(p_f, axis=3)
            acc += np.array(p_f)

        return acc / 8.0

    for idx, fname in enumerate(test_files):
        img_id = os.path.splitext(fname)[0]
        full_img = np.array(Image.open(os.path.join(test_dir, fname)).convert('L'), dtype=np.float32) / 255.0
        H, W = full_img.shape
        pad = max(stride, window)
        padded = cv2.copyMakeBorder(full_img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        ph, pw = padded.shape

        ys = list(range(0, ph - window + 1, stride))
        if ys[-1] != ph - window:
            ys.append(ph - window)
        xs = list(range(0, pw - window + 1, stride))
        if xs[-1] != pw - window:
            xs.append(pw - window)

        chunks, coords = [], []
        for yy in ys:
            for xx in xs:
                chunks.append(padded[yy:yy+window, xx:xx+window])
                coords.append((yy, xx))

        chunks = np.array(chunks, dtype=np.float32)[:, np.newaxis, :, :]
        preds = []
        INF_BATCH = 16
        for i in range(0, len(chunks), INF_BATCH):
            b = jnp.array(chunks[i:i+INF_BATCH], dtype=jnp.bfloat16)
            preds.append(tta_predict(b, len(b)))
        preds = np.concatenate(preds, axis=0)

        full_mask = np.zeros((ph, pw), dtype=np.float32)
        wsum = np.zeros((ph, pw), dtype=np.float32)
        for i, (yy, xx) in enumerate(coords):
            full_mask[yy:yy+window, xx:xx+window] += preds[i][0] * ww
            wsum[yy:yy+window, xx:xx+window] += ww

        full_mask = full_mask / np.maximum(wsum, 1e-5)
        full_mask = full_mask[pad:pad+H, pad:pad+W]
        score_map = cv2.GaussianBlur(full_mask, (5, 5), 1.0)
        binary = (score_map > POS_THR).astype(np.uint8)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        candidates = []
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < MIN_AREA:
                continue
            m = (labels == i)
            if area > SPLIT_AREA:
                wsl = watershed_labels(m)
                for lv in np.unique(wsl):
                    if lv == 0:
                        continue
                    mm = (wsl == lv)
                    if mm.sum() >= MIN_AREA:
                        candidates.append((float(score_map[mm].mean()), mm))
            else:
                candidates.append((float(score_map[m].mean()), m))

        candidates.sort(key=lambda t: t[0], reverse=True)
        claimed = np.zeros_like(binary, dtype=bool)
        kept = []
        for sc, mm in candidates:
            d = mm & ~claimed
            if d.sum() >= MIN_AREA:
                kept.append(d)
                claimed |= d

        for k, mm in enumerate(kept, start=1):
            rle = maskUtils.encode(np.asfortranarray(mm.astype(np.uint8)))
            results.append({'filament_id': f"{img_id}_{k}", 'segmentation_rle': rle['counts'].decode('utf-8')})

        if idx % 20 == 0:
            print(f"{idx+1}/{len(test_files)} | {fname} | {len(kept)} inst")

    out_csv = "/kaggle/working/submission.csv" if IS_KAGGLE else "submission.csv"
    pd.DataFrame(results, columns=['filament_id', 'segmentation_rle']).to_csv(out_csv, index=False)
    print(f"Saved {out_csv} | {len(results)} rows")

if __name__ == "__main__":
    writer = SummaryWriter(LOG_DIR)

    key = jax.random.PRNGKey(SEED)
    steps_per_epoch = max(1, len(train_dataset) // GLOBAL_BATCH_SIZE)
    total_steps = max(1, steps_per_epoch * EPOCHS)

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR,
        warmup_steps=min(WARMUP, total_steps // 10),
        decay_steps=total_steps, end_value=1e-5,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))

    model = UNet(1, 1, base=BASE_CH, key=key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    ema_model = jax.tree_util.tree_map(lambda x: x, model)

    model, ema_model = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (NUM_DEVICES,) + x.shape) if eqx.is_array(x) else x, (model, ema_model))
    opt_state = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (NUM_DEVICES,) + x.shape) if eqx.is_array(x) else x, opt_state)

    train_step = make_train_step(optimizer, EMA_DECAY)

    print(f"Epochs {EPOCHS} | steps/epoch {steps_per_epoch} | total {total_steps} | batch {GLOBAL_BATCH_SIZE}")

    best_val = float('inf')
    saved_ckpts = []
    ema_single = unbroadcast(ema_model)
    gstep = 0

    for epoch in range(EPOCHS):
        ep_loss = 0.0
        if epoch == 2:
            jax.profiler.start_trace(os.path.join(LOG_DIR, "profiler"))

        for batch_idx, (imgs, masks) in enumerate(train_loader):
            key, subkey = jax.random.split(key)
            keys = jax.random.split(subkey, NUM_DEVICES * PER_DEVICE_BATCH).reshape(NUM_DEVICES, PER_DEVICE_BATCH, -1)

            x_in = jnp.array(imgs.numpy().reshape(NUM_DEVICES, PER_DEVICE_BATCH, IMG_SIZE, IMG_SIZE))
            y_in = jnp.array(masks.numpy().reshape(NUM_DEVICES, PER_DEVICE_BATCH, IMG_SIZE, IMG_SIZE))

            x, y = preprocess_train(x_in, y_in, keys)

            model, ema_model, opt_state, loss = train_step(model, ema_model, opt_state, x, y)
            lv = float(jnp.mean(loss))
            ep_loss += lv
            writer.add_scalar('Loss/train_batch', lv, gstep)
            gstep += 1
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx} Loss {lv:.4f}")
            if epoch == 2 and batch_idx == 5:
                jax.profiler.stop_trace()

        ema_single = unbroadcast(ema_model)
        val_loss = validate(ema_single)
        tl = ep_loss / max(1, len(train_loader))
        print(f"Epoch {epoch+1} | Train {tl:.4f} | Val {val_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', tl, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)

        ck = os.path.join(CHECKPOINT_DIR, f"ema_epoch_{epoch+1}.eqx")
        eqx.tree_serialise_leaves(ck, ema_single)
        saved_ckpts.append(ck)
        if len(saved_ckpts) > 3:
            old = saved_ckpts.pop(0)
            if os.path.exists(old):
                os.remove(old)

        if val_loss < best_val:
            best_val = val_loss
            eqx.tree_serialise_leaves(os.path.join(CHECKPOINT_DIR, "ema_best.eqx"), ema_single)
            print(f"New best: {best_val:.4f}")

    writer.close()

    bp = os.path.join(CHECKPOINT_DIR, "ema_best.eqx")
    best_model = eqx.tree_deserialise_leaves(bp, ema_single) if os.path.exists(bp) else ema_single
    generate_submission(best_model)
