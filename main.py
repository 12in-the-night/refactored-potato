import os

# --- KAGGLE TPU OPTIMIZATION: Set XLA memory flags BEFORE importing JAX ---
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true" # Prevents memory fragmentation
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" # Efficient memory release
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Hides TensorFlow warnings (JAX uses TF dataloaders sometimes)

import yaml
import json
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

# --- CPU OPTIMIZATION: Stop OpenCV from blocking PyTorch workers ---
cv2.setNumThreads(0)

# ============================================================
# 1. LOAD YAML CONFIG & DETECT ENVIRONMENT
# ============================================================
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

KAGGLE_BASE = cfg['paths']['kaggle_base']
LOCAL_BASE = cfg['paths']['local_base']
BASE = KAGGLE_BASE if os.path.exists(KAGGLE_BASE) else LOCAL_BASE

IMG_SIZE = cfg['hyperparameters']['img_size']
PER_DEVICE_BATCH = cfg['hyperparameters']['per_device_batch']
EPOCHS = cfg['hyperparameters']['epochs']
LR = cfg['hyperparameters']['learning_rate']
SEED = cfg['hyperparameters']['seed']

NUM_DEVICES = jax.local_device_count()
GLOBAL_BATCH_SIZE = PER_DEVICE_BATCH * NUM_DEVICES
print(f"Environment: {'Kaggle' if BASE == KAGGLE_BASE else 'Local'} | Devices: {NUM_DEVICES} | Batch: {GLOBAL_BATCH_SIZE}")

# ============================================================
# 2. DATASET (ZERO-HICCUP NUMPY MEMORY MAPPING)
# ============================================================
class SolarFilamentDataset(Dataset):
    def __init__(self, chunk_dir, augment=False):
        self.chunk_dir = chunk_dir
        self.augment = augment
        self.files = sorted([f for f in os.listdir(chunk_dir) if f.startswith('img_') and f.endswith('.npy')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = self.files[idx]
        idx_str = img_file.replace('img_', '').replace('.npy', '')
        mask_file = f"mask_{idx_str}.npy"

        # --- KAGGLE OPTIMIZATION: mmap_mode='r' ---
        # This memory-maps the file instead of reading it fully into RAM.
        # It is incredibly fast and prevents RAM exhaustion.
        img = np.load(os.path.join(self.chunk_dir, img_file), mmap_mode='r').copy()
        mask = np.load(os.path.join(self.chunk_dir, mask_file), mmap_mode='r').copy()

        if self.augment:
            # 1. Flips & Rotations
            if np.random.rand() > 0.5: img, mask = np.fliplr(img), np.fliplr(mask)
            if np.random.rand() > 0.5: img, mask = np.flipud(img), np.flipud(mask)
            if np.random.rand() > 0.5: img, mask = np.rot90(img).copy(), np.rot90(mask).copy()

            # 2. Fast Photometric (Gamma + Noise)
            if np.random.rand() > 0.5:
                gamma = np.random.uniform(0.7, 1.3)
                img = np.clip(img ** gamma, 0, 1)
            if np.random.rand() > 0.5:
                noise = np.random.randn(*img.shape).astype(np.float32) * 0.05
                img = np.clip(img + img * noise, 0, 1)

            # 3. Fast Dropout
            if np.random.rand() > 0.3:
                for _ in range(np.random.randint(1, 4)):
                    drop_size = np.random.randint(16, 48)
                    dx = np.random.randint(0, IMG_SIZE - drop_size)
                    dy = np.random.randint(0, IMG_SIZE - drop_size)
                    img[dy:dy+drop_size, dx:dx+drop_size] = 0.0
                    mask[dy:dy+drop_size, dx:dx+drop_size] = 0.0

            # 4. Elastic (Keep probability low to save CPU cycles)
            if np.random.rand() > 0.85: # 15% chance
                alpha = IMG_SIZE * 0.05
                sigma = IMG_SIZE * 0.05
                dx = ndimage.gaussian_filter((np.random.rand(*img.shape).astype(np.float32) * 2 - 1), sigma) * alpha
                dy = ndimage.gaussian_filter((np.random.rand(*img.shape).astype(np.float32) * 2 - 1), sigma) * alpha
                x_grid, y_grid = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
                x_map = (x_grid + dx).astype(np.float32)
                y_map = (y_grid + dy).astype(np.float32)
                img = cv2.remap(img, x_map, y_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                mask = cv2.remap(mask, x_map, y_map, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                mask = (mask > 0.5).astype(np.float32)

        return np.expand_dims(img, axis=0), np.expand_dims(mask, axis=0)

# Ensure this points to your Kaggle dataset
CHUNK_DIR = "/kaggle/input/solar-chunks/preprocessed_chunks"
train_dataset = SolarFilamentDataset(CHUNK_DIR, augment=True)

# --- KAGGLE OPTIMIZATION: Persistent Workers & Prefetch ---
# We use 4 workers (Kaggle TPU VMs have 4 vCPUs)
train_loader = DataLoader(
    train_dataset,
    batch_size=GLOBAL_BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=4,               # Use all 4 Kaggle vCPUs
    persistent_workers=True,     # Stops workers from dying between epochs (huge time saver)
    prefetch_factor=4,           # Preloads 16 batches into RAM so TPU never waits
    pin_memory=False             # Must be False for TPU
)

# ============================================================
# 3. EQUINOX U-NET MODEL
# ============================================================
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
    down1: ConvBlock; down2: ConvBlock; down3: ConvBlock
    pool: eqx.nn.MaxPool2d
    up_conv3: eqx.nn.ConvTranspose2d; up_conv2: eqx.nn.ConvTranspose2d; up_conv1: eqx.nn.ConvTranspose2d
    up1: ConvBlock; up2: ConvBlock; up3: ConvBlock
    out_conv: eqx.nn.Conv2d
    def __init__(self, in_channels=1, out_channels=1, key=jax.random.PRNGKey(0)):
        keys = jax.random.split(key, 11)
        self.pool = eqx.nn.MaxPool2d(2, 2)
        self.down1 = ConvBlock(in_channels, 16, keys[0])
        self.down2 = ConvBlock(16, 32, keys[1])
        self.down3 = ConvBlock(32, 64, keys[2])
        self.up_conv3 = eqx.nn.ConvTranspose2d(64, 32, 2, 2, key=keys[3])
        self.up3 = ConvBlock(64, 32, keys[4])
        self.up_conv2 = eqx.nn.ConvTranspose2d(32, 16, 2, 2, key=keys[5])
        self.up2 = ConvBlock(32, 16, keys[6])
        self.up_conv1 = eqx.nn.ConvTranspose2d(16, 8, 2, 2, key=keys[7])
        self.up1 = ConvBlock(16, 8, keys[8])
        self.out_conv = eqx.nn.Conv2d(8, out_channels, 1, key=keys[9])
    def __call__(self, x):
        x1 = self.down1(x); p1 = self.pool(x1)
        x2 = self.down2(p1); p2 = self.pool(x2)
        x3 = self.down3(p2); p3 = self.pool(x3)
        u3 = self.up_conv3(p3); x3_up = self.up3(jnp.concatenate([u3, x3], axis=0))
        u2 = self.up_conv2(x3_up); x2_up = self.up2(jnp.concatenate([u2, x2], axis=0))
        u1 = self.up_conv1(x2_up); x1_up = self.up1(jnp.concatenate([u1, x1], axis=0))
        return self.out_conv(x1_up)

# ============================================================
# 4. ELITE LOSS FUNCTIONS
# ============================================================
def focal_loss(preds, targets, alpha=0.8, gamma=2.0):
    bce = optax.sigmoid_binary_cross_entropy(preds, targets)
    pt = jnp.exp(-bce)
    return jnp.mean(alpha * (1 - pt)**gamma * bce)

def dice_loss(preds, targets):
    preds = jax.nn.sigmoid(preds).reshape(-1); targets = targets.reshape(-1)
    intersection = jnp.sum(preds * targets)
    union = jnp.sum(preds) + jnp.sum(targets)
    return 1.0 - (2. * intersection + 1e-5) / (union + 1e-5)

def boundary_loss(preds, targets):
    t_edges = jnp.abs(jnp.roll(targets, 1, axis=2) - targets) + jnp.abs(jnp.roll(targets, 1, axis=3) - targets)
    t_edges = jnp.clip(t_edges, 0, 1)
    bce = optax.sigmoid_binary_cross_entropy(preds, targets)
    return jnp.mean(bce * (1.0 + 5.0 * t_edges))

# ============================================================
# 5. TPU TRAINING STEP (WITH EMA)
# ============================================================
@eqx.filter_pmap(axis_name='batch')
def train_step(model, ema_model, opt_state, x, y, optimizer, ema_decay=0.999):
    def loss_fn(model, x, y):
        preds = jax.vmap(model)(x)
        return focal_loss(preds, y) + dice_loss(preds, y) + boundary_loss(preds, y)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    grads = jax.lax.pmean(grads, axis_name='batch')

    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)

    ema_model = jax.tree_util.tree_map(lambda e, m: ema_decay * e + (1 - ema_decay) * m, ema_model, model)

    return model, ema_model, opt_state, jax.lax.pmean(loss, axis_name='batch')

# ============================================================
# 6. POST-PROCESSING & INFERENCE
# ============================================================
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def elite_watershed(binary_mask):
    dist = ndimage.distance_transform_edt(binary_mask)
    local_max = h_maxima(dist, h=1.5)
    markers = ndimage.label(local_max)[0].astype(np.int32)
    labels = watershed(-dist, markers, mask=binary_mask)
    return (labels > 0).astype(np.uint8)

def generate_submission(model):
    print("Generating ELITE Submission (Batched Inference)...")
    test_img_dir = os.path.join(BASE, cfg['paths']['test_img'])
    test_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith('.jpeg')])
    results = []

    batch_size_inference = 32

    @eqx.filter_jit
    def predict_batch(imgs):
        return jax.nn.sigmoid(jax.vmap(model)(imgs))

    hann_1d = np.hanning(IMG_SIZE)
    window_weight = np.maximum(np.outer(hann_1d, hann_1d), 0.01)

    stride = 256; window = IMG_SIZE; pad_size = 256

    for idx, fname in enumerate(test_files):
        full_img = Image.open(os.path.join(test_img_dir, fname)).convert('L')
        full_img = np.array(full_img, dtype=np.float32) / 255.0

        padded_img = cv2.copyMakeBorder(full_img, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
        padded_size = padded_img.shape[0]

        full_mask = np.zeros((padded_size, padded_size), dtype=np.float32)
        weight_sum = np.zeros((padded_size, padded_size), dtype=np.float32)

        chunks = []
        coords = []
        for y in range(0, padded_size - window + 1, stride):
            for x in range(0, padded_size - window + 1, stride):
                chunks.append(padded_img[y:y+window, x:x+window])
                coords.append((y, x))

        chunks = np.array(chunks)[:, np.newaxis, :, :]

        preds = []
        for i in range(0, len(chunks), batch_size_inference):
            batch_chunks = jnp.array(chunks[i:i+batch_size_inference])
            p1 = np.array(predict_batch(batch_chunks))
            p2 = np.flip(np.array(predict_batch(jnp.flip(batch_chunks, axis=3))), axis=3)
            preds.append((p1 + p2) / 2.0)

        preds = np.concatenate(preds, axis=0)

        for i, (y, x) in enumerate(coords):
            full_mask[y:y+window, x:x+window] += preds[i][0] * window_weight
            weight_sum[y:y+window, x:x+window] += window_weight

        full_mask = full_mask / np.maximum(weight_sum, 1e-5)
        full_mask = full_mask[pad_size:pad_size+2048, pad_size:pad_size+2048]

        full_mask = cv2.GaussianBlur(full_mask, (5, 5), 1.0)
        binary_mask = (full_mask > 0.5).astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 50:
                binary_mask[labels == i] = 0

        binary_mask = elite_watershed(binary_mask)
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        results.append({'id': fname.replace('.jpeg', ''), 'segmentation_rle': rle_encode(binary_mask)})
        if idx % 20 == 0: print(f"Processed {idx+1}/{len(test_files)}")

    pd.DataFrame(results).to_csv('submission.csv', index=False)
    print("Submission saved successfully!")

# ============================================================
# 7. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    key = jax.random.PRNGKey(SEED)

    # Calculate steps based on the new preprocessed dataset size
    total_steps = (len(train_dataset) // GLOBAL_BATCH_SIZE) * EPOCHS
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR, warmup_steps=100, decay_steps=total_steps, end_value=1e-5
    )
    optimizer = optax.adam(learning_rate=schedule)

    model = UNet(in_channels=1, out_channels=1, key=key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    ema_model = jax.tree_util.tree_map(lambda x: x, model)

    model, ema_model = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (NUM_DEVICES,) + x.shape) if eqx.is_array(x) else x, (model, ema_model))
    opt_state = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (NUM_DEVICES,) + x.shape) if eqx.is_array(x) else x, opt_state)

    print(f"Starting ELITE training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            x = jnp.array(imgs.numpy()).reshape(NUM_DEVICES, PER_DEVICE_BATCH, 1, IMG_SIZE, IMG_SIZE)
            y = jnp.array(masks.numpy()).reshape(NUM_DEVICES, PER_DEVICE_BATCH, 1, IMG_SIZE, IMG_SIZE)

            model, ema_model, opt_state, loss = train_step(model, ema_model, opt_state, x, y, optimizer)
            epoch_loss += jnp.mean(loss)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {jnp.mean(loss):.4f}")

        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(train_loader):.4f}")

    print("Training complete. Extracting EMA Shadow Model for inference...")
    ema_model = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, ema_model)
    generate_submission(ema_model)
