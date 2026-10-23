import os
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

cv2.setNumThreads(0)

LOCAL_BASE = "./data/MAGFiLO_1.0_Kaggle_2026"
IMG_DIR = os.path.join(LOCAL_BASE, "train/train_images")
ANN_FILE = os.path.join(LOCAL_BASE, "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
OUT_DIR = "./data/preprocessed_chunks"
VAL_DIR = "./data/preprocessed_chunks_val"

IMG_SIZE = 512
VAL_SPLIT = 0.1
NUM_WORKERS = os.cpu_count()
SEED = 13

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

with open(ANN_FILE, 'r') as f:
    data = json.load(f)

ann_index = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in ann_index: ann_index[img_id] = []
    ann_index[img_id].append(ann)

file_to_id = {img['file_name']: img['id'] for img in data['images']}

def process_image(args):
    idx, fname, is_val = args
    if not fname.endswith('.jpeg'): return 0

    # Seed RNG per worker based on image index to prevent repetitive crops
    np.random.seed(idx + SEED)

    img_path = os.path.join(IMG_DIR, fname)
    img = Image.open(img_path).convert('L')
    img = np.array(img, dtype=np.float32) / 255.0

    mask = np.zeros((2048, 2048), dtype=np.float32)
    img_id = file_to_id.get(fname)

    if img_id in ann_index:
        for ann in ann_index[img_id]:
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape(-1, 2)
                    cv2.fillPoly(mask, [poly.astype(np.int32)], 1.0)

    out_dir = VAL_DIR if is_val else OUT_DIR
    chunks_saved = 0

    if np.sum(mask) > 0:
        y_coords, x_coords = np.where(mask == 1)
        num_crops = min(4, len(y_coords))
        indices = np.random.choice(len(y_coords), num_crops, replace=False)

        for i in indices:
            cy, cx = y_coords[i], x_coords[i]
            x = np.clip(cx - IMG_SIZE//2, 0, 2048 - IMG_SIZE)
            y = np.clip(cy - IMG_SIZE//2, 0, 2048 - IMG_SIZE)

            img_chunk = img[y:y+IMG_SIZE, x:x+IMG_SIZE]
            mask_chunk = mask[y:y+IMG_SIZE, x:x+IMG_SIZE]

            np.save(os.path.join(out_dir, f"img_{idx}_{chunks_saved}.npy"), img_chunk)
            np.save(os.path.join(out_dir, f"mask_{idx}_{chunks_saved}.npy"), mask_chunk)
            chunks_saved += 1

    x = np.random.randint(0, 2048 - IMG_SIZE)
    y = np.random.randint(0, 2048 - IMG_SIZE)
    np.save(os.path.join(out_dir, f"img_{idx}_{chunks_saved}.npy"), img[y:y+IMG_SIZE, x:x+IMG_SIZE])
    np.save(os.path.join(out_dir, f"mask_{idx}_{chunks_saved}.npy"), mask[y:y+IMG_SIZE, x:x+IMG_SIZE])

    return 1

if __name__ == "__main__":
    print(f"Starting Pre-processing with {NUM_WORKERS} CPU workers...")
    all_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpeg')])

    # Split by image index, not by chunk, to prevent train/val leakage
    np.random.seed(SEED)
    val_indices = set(np.random.choice(len(all_files), int(len(all_files) * VAL_SPLIT), replace=False))

    tasks = [(i, f, i in val_indices) for i, f in enumerate(all_files)]

    total_chunks = 0
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for result in tqdm(executor.map(process_image, tasks), total=len(tasks), desc="Processing Images"):
            total_chunks += result

    print(f"\nSuccess! Generated {total_chunks} chunks.")
