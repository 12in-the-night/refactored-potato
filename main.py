import os
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

# ============================================================
# 1. LOAD YAML CONFIG & DETECT ENVIRONMENT
# ============================================================
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

KAGGLE_BASE = cfg['paths']['kaggle_base']
LOCAL_BASE = cfg['paths']['local_base']

if os.path.exists(KAGGLE_BASE):
    print("Kaggle environment detected.")
    BASE = KAGGLE_BASE
else:
    print("Local PC environment detected.")
    BASE = LOCAL_BASE

TRAIN_IMG_DIR = os.path.join(BASE, cfg['paths']['train_img'])
ANN_FILE = os.path.join(BASE, cfg['paths']['ann_file'])
TEST_IMG_DIR = os.path.join(BASE, cfg['paths']['test_img'])

IMG_SIZE = cfg['hyperparameters']['img_size'] # 512
PER_DEVICE_BATCH = cfg['hyperparameters']['per_device_batch']
EPOCHS = cfg['hyperparameters']['epochs']
LR = cfg['hyperparameters']['learning_rate']
SEED = cfg['hyperparameters']['seed']

NUM_DEVICES = jax.local_device_count()
GLOBAL_BATCH_SIZE = PER_DEVICE_BATCH * NUM_DEVICES
print(f"Detected {NUM_DEVICES} device(s). Global Batch Size: {GLOBAL_BATCH_SIZE}")

# ============================================================
# 2. DATASET (WITH RANDOM CROPPING FOR TRAINING)
# ============================================================
class SolarFilamentDataset(Dataset):
    def __init__(self, img_dir, ann_file=None, img_size=512):
        self.img_dir = img_dir
        self.img_size = img_size
        self.ann_file = ann_file # Save this to check training mode later
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpeg')])

        self.ann_index = {}
        if self.ann_file:
            with open(self.ann_file, 'r') as f:
                data = json.load(f)
            for ann in data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.ann_index:
                    self.ann_index[img_id] = []
                self.ann_index[img_id].append(ann)
            self.file_to_id = {img['file_name']: img['id'] for img in data['images']}

    def __len__(self):
        return len(self.img_files)

    def _poly_to_mask(self, anns, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape(-1, 2)
                    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        return mask

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        img_path = os.path.join(self.img_dir, fname)

        # Load FULL resolution image (2048x2048)
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32) / 255.0

        mask = np.zeros((2048, 2048), dtype=np.float32)
        if self.ann_file:
            img_id = self.file_to_id.get(fname)
            if img_id in self.ann_index:
                mask = self._poly_to_mask(self.ann_index[img_id], 2048, 2048)
                mask = mask.astype(np.float32)

        # CHUNKING: Random Crop for Training
        if self.ann_file:
            # Pick a random top-left corner
            x = np.random.randint(0, 2048 - self.img_size)
            y = np.random.randint(0, 2048 - self.img_size)

            img = img[y:y+self.img_size, x:x+self.img_size]
            mask = mask[y:y+self.img_size, x:x+self.img_size]
        else:
            # For testing, we just resize (but we bypass this in sliding window inference)
            img = cv2.resize(img, (self.img_size, self.img_size))
            mask = cv2.resize(mask, (self.img_size, self.img_size))

        img = np.expand_dims(img, axis=0)  # (1, 512, 512)
        mask = np.expand_dims(mask, axis=0) # (1, 512, 512)
        return img, mask

train_dataset = SolarFilamentDataset(TRAIN_IMG_DIR, ANN_FILE, IMG_SIZE)
train_loader = DataLoader(train_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

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
        self.conv1 = eqx.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, key=k2)
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
    pool: eqx.nn.MaxPool2d
    up_conv3: eqx.nn.ConvTranspose2d
    up_conv2: eqx.nn.ConvTranspose2d
    up_conv1: eqx.nn.ConvTranspose2d
    up1: ConvBlock
    up2: ConvBlock
    up3: ConvBlock
    out_conv: eqx.nn.Conv2d

    def __init__(self, in_channels=1, out_channels=1, key=jax.random.PRNGKey(0)):
        keys = jax.random.split(key, 11)
        self.pool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)

        self.down1 = ConvBlock(in_channels, 16, keys[0])
        self.down2 = ConvBlock(16, 32, keys[1])
        self.down3 = ConvBlock(32, 64, keys[2])

        self.up_conv3 = eqx.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, key=keys[3])
        self.up3 = ConvBlock(64, 32, keys[4])

        self.up_conv2 = eqx.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, key=keys[5])
        self.up2 = ConvBlock(32, 16, keys[6])

        self.up_conv1 = eqx.nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, key=keys[7])
        self.up1 = ConvBlock(16, 8, keys[8])

        self.out_conv = eqx.nn.Conv2d(8, out_channels, kernel_size=1, key=keys[9])

    def __call__(self, x):
        x1 = self.down1(x)
        p1 = self.pool(x1)
        x2 = self.down2(p1)
        p2 = self.pool(x2)
        x3 = self.down3(p2)
        p3 = self.pool(x3)

        u3 = self.up_conv3(p3)
        concat3 = jnp.concatenate([u3, x3], axis=0)
        x3_up = self.up3(concat3)

        u2 = self.up_conv2(x3_up)
        concat2 = jnp.concatenate([u2, x2], axis=0)
        x2_up = self.up2(concat2)

        u1 = self.up_conv1(x2_up)
        concat1 = jnp.concatenate([u1, x1], axis=0)
        x1_up = self.up1(concat1)

        return self.out_conv(x1_up)

# ============================================================
# 4. LOSS & TPU TRAINING STEP
# ============================================================
def dice_loss(preds, targets):
    preds = jax.nn.sigmoid(preds).reshape(-1)
    targets = targets.reshape(-1)
    smooth = 1e-5
    intersection = jnp.sum(preds * targets)
    union = jnp.sum(preds) + jnp.sum(targets)
    return 1.0 - (2. * intersection + smooth) / (union + smooth)

def bce_loss(preds, targets):
    return optax.sigmoid_binary_cross_entropy(preds, targets).mean()

def total_loss(preds, targets):
    return bce_loss(preds, targets) + dice_loss(preds, targets)

@eqx.filter_pmap(axis_name='batch')
def train_step(model, opt_state, x, y, optimizer):
    def loss_fn(model, x, y):
        preds = jax.vmap(model)(x)
        return total_loss(preds, y)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    grads = jax.lax.pmean(grads, axis_name='batch')

    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)

    loss = jax.lax.pmean(loss, axis_name='batch')
    return model, opt_state, loss

# ============================================================
# 5. SLIDING WINDOW INFERENCE & SUBMISSION
# ============================================================
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def generate_submission(model):
    print("Generating submission using Sliding Window (Overlap) with Morphological Closing...")
    test_ds = SolarFilamentDataset(TEST_IMG_DIR, ann_file=None, img_size=IMG_SIZE)
    results = []

    @eqx.filter_jit
    def predict(img):
        return jax.nn.sigmoid(model(img))

    for idx in range(len(test_ds)):
        fname = test_ds.img_files[idx]
        img_path = os.path.join(TEST_IMG_DIR, fname)

        # 1. Load the FULL 2048x2048 image
        full_img = Image.open(img_path).convert('L')
        full_img = np.array(full_img, dtype=np.float32) / 255.0

        # 2. Setup Sliding Window variables
        full_mask = np.zeros((2048, 2048), dtype=np.float32)
        count_mask = np.zeros((2048, 2048), dtype=np.float32)

        stride = 256 # 50% overlap
        window = IMG_SIZE # 512

        # 3. Slide across the image
        for y in range(0, 2048 - window + 1, stride):
            for x in range(0, 2048 - window + 1, stride):
                chunk = full_img[y:y+window, x:x+window]
                chunk_input = jnp.expand_dims(jnp.array(chunk), axis=0)

                pred = predict(chunk_input)
                pred = np.array(pred)[0]

                full_mask[y:y+window, x:x+window] += pred
                count_mask[y:y+window, x:x+window] += 1.0

        # 4. Average the overlapping predictions
        full_mask = full_mask / np.maximum(count_mask, 1e-5)

        # 5. Threshold to binary
        binary_mask = (full_mask > 0.5).astype(np.uint8)

        # 6. Morphological Closing (Bridges 1-2 pixel gaps at seams)
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # 7. RLE Encode
        rle = rle_encode(binary_mask)
        results.append({'id': fname.replace('.jpeg', ''), 'segmentation_rle': rle})

        if idx % 20 == 0:
            print(f"Processed {idx+1}/{len(test_ds)} images")

    df = pd.DataFrame(results)
    df.to_csv('submission.csv', index=False)
    print("Submission saved successfully!")

# ============================================================
# 6. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    key = jax.random.PRNGKey(SEED)

    model = UNet(in_channels=1, out_channels=1, key=key)
    optimizer = optax.adam(learning_rate=LR)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Replicate model across TPU cores
    model = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (NUM_DEVICES,) + x.shape) if eqx.is_array(x) else x, model)
    opt_state = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (NUM_DEVICES,) + x.shape) if eqx.is_array(x) else x, opt_state)

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (imgs, masks) in enumerate(train_loader):
            # Reshape for TPU (Global_Batch -> 8 cores * Per_Device_Batch)
            x = jnp.array(imgs.numpy()).reshape(NUM_DEVICES, PER_DEVICE_BATCH, 1, IMG_SIZE, IMG_SIZE)
            y = jnp.array(masks.numpy()).reshape(NUM_DEVICES, PER_DEVICE_BATCH, 1, IMG_SIZE, IMG_SIZE)

            model, opt_state, loss = train_step(model, opt_state, x, y, optimizer)

            epoch_loss += jnp.mean(loss)
            num_batches += 1

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {jnp.mean(loss):.4f}")

        print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss / num_batches:.4f}")

    # Collapse model back to single state for inference
    model = jax.tree_util.tree_map(lambda x: x[0] if eqx.is_array(x) else x, model)
    generate_submission(model)
