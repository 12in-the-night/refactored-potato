import os
import glob
import json
import numpy as np
import cv2
import pandas as pd
from scipy import ndimage
from skimage.morphology import h_maxima
from skimage.segmentation import watershed
from pycocotools import mask as maskUtils
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

cv2.setNumThreads(0)
jax.config.update("jax_default_matmul_precision", "bfloat16")

# --- 1. DATASET ---
class SolarDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=512):
        self.img_dir = img_dir
        self.img_size = img_size
        with open(ann_file, 'r') as f:
            data = json.load(f)
        self.images = data['images']
        self.ann_index = {}
        for ann in data['annotations']:
            self.ann_index.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        fname = img_info['file_name']
        img_id = img_info['id']

        img_path = os.path.join(self.img_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return np.zeros((1, self.img_size, self.img_size), dtype=np.float32), \
                   np.zeros((1, self.img_size, self.img_size), dtype=np.float32)

        mask = np.zeros(img.shape, dtype=np.uint8)
        if img_id in self.ann_index:
            for ann in self.ann_index[img_id]:
                if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                    for seg in ann['segmentation']:
                        poly = np.array(seg, dtype=np.float32).reshape(-1, 2)
                        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        return (img[None, ...].astype(np.float32) / 255.0), mask[None, ...].astype(np.float32)

# --- 2. MODEL ---
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
        return jax.nn.relu(self.norm2(self.conv2(x)))

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
    def __init__(self, in_channels=1, out_channels=1, base=32, key=jax.random.PRNGKey(0)):
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

# --- 3. LOSSES ---
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

def make_train_step(optimizer):
    def loss_fn(model, x, y):
        model_bf16 = jax.tree_util.tree_map(lambda p: p.astype(jnp.bfloat16) if eqx.is_array(p) else p, model)
        preds = jax.vmap(model_bf16)(x)
        return 0.2 * focal_loss(preds, y) + 0.6 * dice_loss(preds, y) + 0.2 * boundary_loss(preds, y)

    @eqx.filter_pmap(axis_name='batch')
    def train_step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
        grads = jax.tree_util.tree_map(lambda g: jax.lax.pmean(g, axis_name='batch'), grads)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, jax.lax.pmean(loss, axis_name='batch')
    return train_step

# --- 4. POST-PROCESSING ---
def watershed_labels(binary_mask, ws_h=1.0):
    dist = ndimage.distance_transform_edt(binary_mask)
    local_max = h_maxima(dist, h=ws_h)
    markers = ndimage.label(local_max)[0].astype(np.int32)
    cc, _ = ndimage.label(binary_mask)
    markers = np.where((markers == 0) & (binary_mask > 0), cc + markers.max() + 1, markers)
    return watershed(-dist, markers, mask=binary_mask).astype(np.int32)

def generate_submission(model, test_dir, out_csv, img_size=512):
    test_files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
    results = []
    model_bf16 = jax.tree_util.tree_map(lambda p: p.astype(jnp.bfloat16) if eqx.is_array(p) else p, model)

    @eqx.filter_jit
    def predict_batch(m, imgs):
        return jax.nn.sigmoid(jax.vmap(m)(imgs))

    print("Generating submission...")
    for idx, fname in enumerate(tqdm(test_files)):
        img_id = os.path.splitext(fname)[0]
        img_orig = cv2.imread(os.path.join(test_dir, fname), cv2.IMREAD_GRAYSCALE)
        H, W = img_orig.shape
        img = cv2.resize(img_orig, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        inp = jnp.array(img[None, None, ...], dtype=jnp.bfloat16)
        pred = np.array(predict_batch(model_bf16, inp))[0, 0]

        pred_mask = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
        score_map = cv2.GaussianBlur(pred_mask, (5, 5), 1.0)
        binary = (score_map > 0.5).astype(np.uint8)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        candidates = []
        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 40: continue
            m = (labels == i)
            if area > 3500:
                wsl = watershed_labels(m)
                for lv in np.unique(wsl):
                    if lv == 0: continue
                    mm = (wsl == lv)
                    if mm.sum() >= 40: candidates.append((float(score_map[mm].mean()), mm))
            else:
                candidates.append((float(score_map[m].mean()), m))

        candidates.sort(key=lambda t: t[0], reverse=True)
        claimed = np.zeros_like(binary, dtype=bool)
        kept = []
        for sc, mm in candidates:
            d = mm & ~claimed
            if d.sum() >= 40:
                kept.append(d)
                claimed |= d

        for k, mm in enumerate(kept, start=1):
            rle = maskUtils.encode(np.asfortranarray(mm.astype(np.uint8)))
            results.append({'filament_id': f"{img_id}_{k}", 'segmentation_rle': rle['counts'].decode('utf-8')})

    pd.DataFrame(results, columns=['filament_id', 'segmentation_rle']).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

# --- 5. MAIN ---
if __name__ == "__main__":
    IS_KAGGLE = os.path.exists("/kaggle/input")

    if IS_KAGGLE:
        # Auto-find the JSON file anywhere in /kaggle/input
        ann_search = glob.glob("/kaggle/input/**/MAGFiLO_1.0_Annotations_kaggle2026_train.json", recursive=True)
        if not ann_search:
            raise FileNotFoundError("Could not find dataset JSON. Ensure competition dataset is added.")
        ANN_FILE = ann_search[0]
        BASE = os.path.dirname(os.path.dirname(ANN_FILE))
        IMG_DIR = os.path.join(os.path.dirname(ANN_FILE), "train_images")
        TEST_DIR = os.path.join(BASE, "test/test_images")
    else:
        BASE = "./data/MAGFiLO_1.0_Kaggle_2026"
        ANN_FILE = os.path.join(BASE, "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
        IMG_DIR = os.path.join(BASE, "train/train_images")
        TEST_DIR = os.path.join(BASE, "test/test_images")

    print(f"Dataset found at: {BASE}")

    IMG_SIZE = 512
    BATCH_SIZE = 16
    EPOCHS = 20
    LR = 0.0008
    BASE_CH = 32

    NUM_DEVICES = jax.local_device_count()
    GLOBAL_BATCH = BATCH_SIZE * NUM_DEVICES

    dataset = SolarDataset(IMG_DIR, ANN_FILE, img_size=IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=GLOBAL_BATCH, shuffle=True, num_workers=4, drop_last=True)

    key = jax.random.PRNGKey(13)
    schedule = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=LR, warmup_steps=100, decay_steps=len(dataloader)*EPOCHS, end_value=1e-5)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))

    model = UNet(1, 1, base=BASE_CH, key=key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    model = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (NUM_DEVICES,) + x.shape), model)
    opt_state = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (NUM_DEVICES,) + x.shape), opt_state)

    train_step = make_train_step(optimizer)

    print(f"Start training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        ep_loss = 0.0
        for imgs, masks in dataloader:
            x = jnp.array(imgs.numpy()).reshape(NUM_DEVICES, BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
            y = jnp.array(masks.numpy()).reshape(NUM_DEVICES, BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)
            model, opt_state, loss = train_step(model, opt_state, x, y)
            ep_loss += float(jnp.mean(loss))
        print(f"Epoch {epoch+1} Loss: {ep_loss/len(dataloader):.4f}")

    model_single = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) and x.ndim > 0 else x, model)

    out_weights = "/kaggle/working/ema_best.eqx" if IS_KAGGLE else "ema_best.eqx"
    out_csv = "/kaggle/working/submission.csv" if IS_KAGGLE else "submission.csv"

    eqx.tree_serialise_leaves(out_weights, model_single)
    print(f"Model saved to {out_weights}")

    generate_submission(model_single, TEST_DIR, out_csv, IMG_SIZE)
