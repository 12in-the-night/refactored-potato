import os

if os.path.exists("/dev/accel0") or os.environ.get("TPU_NAME") or os.environ.get("COLAB_TPU_ADDR"):
    os.environ["JAX_PLATFORMS"] = "tpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
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

# num_workers=0 prevents JAX/PyTorch fork deadlock on Kaggle
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
    x = x.astype(jnp.float
