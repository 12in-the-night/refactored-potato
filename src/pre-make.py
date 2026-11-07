# Preprocessing script for solar filament segmentation data
# This script processes raw images and annotations into preprocessed chunks

# Import required modules
import os
import json
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Disable OpenCV multithreading for consistent performance
cv2.setNumThreads(0)

# Check if running in Kaggle environment
IS_KAGGLE = os.path.exists("/kaggle/input")
# Set base path based on environment (Kaggle vs local)
LOCAL_BASE = "/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026" if IS_KAGGLE else "./data/MAGFiLO_1.0_Kaggle_2026"
# Path to training images directory
IMG_DIR = os.path.join(LOCAL_BASE, "train/train_images")
# Path to annotations JSON file
ANN_FILE = os.path.join(LOCAL_BASE, "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
# Output directory for training chunks
OUT_DIR = "/kaggle/working/preprocessed_chunks" if IS_KAGGLE else "./data/preprocessed_chunks"
# Output directory for validation chunks
VAL_DIR = "/kaggle/working/preprocessed_chunks_val" if IS_KAGGLE else "./data/preprocessed_chunks_val"

# Image patch size for chunking
IMG_SIZE = 512
# Fraction of data to use for validation
VAL_SPLIT = 0.1
# Minimum filament area to consider a positive sample
MIN_FILAMENT_AREA = 50
# Number of parallel workers (use all CPU cores)
NUM_WORKERS = os.cpu_count()
# Random seed for reproducibility
SEED = 13

# Create output directories if they don't exist
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Load annotations from JSON file
with open(ANN_FILE, 'r') as f:
    data = json.load(f)

# Build index mapping image_id to list of annotations
ann_index = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    ann_index.setdefault(img_id, []).append(ann)

# Build mapping from filename to image_id
file_to_id = {img['file_name']: img['id'] for img in data['images']}

# Process a single image: extract positive and negative patches
def process_image(args):
    idx, fname, is_val = args
    # Skip non-image files
    if not fname.lower().endswith(('.jpeg', '.jpg', '.png')):
        return 0

    # Set random seed per image for reproducibility
    np.random.seed(idx + SEED)

    # cv2.imread is significantly faster than PIL
    img = cv2.imread(os.path.join(IMG_DIR, fname), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    H, W = img.shape

    # Create empty mask
    mask = np.zeros((H, W), dtype=np.uint8)
    img_id = file_to_id.get(fname)
    if img_id in ann_index:
        for ann in ann_index[img_id]:
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    poly = np.array(seg, dtype=np.float32).reshape(-1, 2)
                    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

    # Choose output directory based on train/val split
    out_dir = VAL_DIR if is_val else OUT_DIR
    saved = 0
    # Find coordinates of filament pixels
    y_coords, x_coords = np.where(mask == 1)

    # Extract positive patches centered on filament pixels
    if len(y_coords) > 0 and H >= IMG_SIZE and W >= IMG_SIZE:
        attempts = 0
        while saved < 4 and attempts < 20:
            attempts += 1
            r = np.random.randint(0, len(y_coords))
            cy, cx = y_coords[r], x_coords[r]
            x = int(np.clip(cx - IMG_SIZE // 2, 0, W - IMG_SIZE))
            y = int(np.clip(cy - IMG_SIZE // 2, 0, H - IMG_SIZE))

            ic = img[y:y+IMG_SIZE, x:x+IMG_SIZE]
            mc = mask[y:y+IMG_SIZE, x:x+IMG_SIZE]

            if int(mc.sum()) > MIN_FILAMENT_AREA:
                np.save(os.path.join(out_dir, f"img_{idx}_{saved}.npy"), ic)
                np.save(os.path.join(out_dir, f"mask_{idx}_{saved}.npy"), mc)
                saved += 1

    # Extract one negative patch (no filament)
    neg_saved = False
    neg_attempts = 0
    if H >= IMG_SIZE and W >= IMG_SIZE:
        while not neg_saved and neg_attempts < 15:
            neg_attempts += 1
            x = np.random.randint(0, W - IMG_SIZE + 1)
            y = np.random.randint(0, H - IMG_SIZE + 1)
            ic = img[y:y+IMG_SIZE, x:x+IMG_SIZE]
            mc = mask[y:y+IMG_SIZE, x:x+IMG_SIZE]
            if int(mc.sum()) == 0:
                np.save(os.path.join(out_dir, f"img_{idx}_{saved}.npy"), ic)
                np.save(os.path.join(out_dir, f"mask_{idx}_{saved}.npy"), mc)
                neg_saved = True

    return saved + (1 if neg_saved else 0)

if __name__ == "__main__":
    print(f"Workers: {NUM_WORKERS}")
    # Get all image files in directory
    all_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
    np.random.seed(SEED)
    # Randomly select validation indices
    val_idx = set(np.random.choice(len(all_files), int(len(all_files) * VAL_SPLIT), replace=False).tolist())
    # Create task list with (index, filename, is_val) tuples
    tasks = [(i, f, i in val_idx) for i, f in enumerate(all_files)]

    total = 0
    # Process images in parallel using process pool
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for r in tqdm(ex.map(process_image, tasks, chunksize=16), total=len(tasks)):
            total += r
    print(f"\nDone. {total} chunks.")