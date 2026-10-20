import os
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# --- CPU OPTIMIZATION ---
cv2.setNumThreads(0) # Prevent OpenCV from fighting Python's multiprocessing

# ============================================================
# 1. CONFIGURATION (Adjust these paths!)
# ============================================================
# Point these to your NVMe drive for maximum speed
LOCAL_BASE = "./data/MAGFiLO_1.0_Kaggle_2026"
IMG_DIR = os.path.join(LOCAL_BASE, "train/train_images")
ANN_FILE = os.path.join(LOCAL_BASE, "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json")
OUT_DIR = "./data/preprocessed_chunks" # Output folder on NVMe

IMG_SIZE = 512
NUM_WORKERS = os.cpu_count() # Use all available CPU cores

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# 2. LOAD ANNOTATIONS
# ============================================================
with open(ANN_FILE, 'r') as f:
    data = json.load(f)

ann_index = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in ann_index:
        ann_index[img_id] = []
    ann_index[img_id].append(ann)

file_to_id = {img['file_name']: img['id'] for img in data['images']}

# ============================================================
# 3. PROCESSING FUNCTION (Runs in parallel)
# ============================================================
def process_image(args):
    idx, fname = args
    if not fname.endswith('.jpeg'): return 0

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

    chunks_saved = 0

    # 1. Filament-Centered Crops
    if np.sum(mask) > 0:
        y_coords, x_coords = np.where(mask == 1)
        num_crops = min(4, len(y_coords)) # Extract up to 4 crops per image
        indices = np.random.choice(len(y_coords), num_crops, replace=False)

        for i in indices:
            cy, cx = y_coords[i], x_coords[i]
            x = np.clip(cx - IMG_SIZE//2, 0, 2048 - IMG_SIZE)
            y = np.clip(cy - IMG_SIZE//2, 0, 2048 - IMG_SIZE)

            img_chunk = img[y:y+IMG_SIZE, x:x+IMG_SIZE]
            mask_chunk = mask[y:y+IMG_SIZE, x:x+IMG_SIZE]

            # Save as separate .npy files (Ultra fast loading)
            np.save(os.path.join(OUT_DIR, f"img_{idx}_{chunks_saved}.npy"), img_chunk)
            np.save(os.path.join(OUT_DIR, f"mask_{idx}_{chunks_saved}.npy"), mask_chunk)
            chunks_saved += 1

    # 2. Random Background Crop (Teaches the model what empty space looks like)
    x = np.random.randint(0, 2048 - IMG_SIZE)
    y = np.random.randint(0, 2048 - IMG_SIZE)
    np.save(os.path.join(OUT_DIR, f"img_{idx}_{chunks_saved}.npy"), img[y:y+IMG_SIZE, x:x+IMG_SIZE])
    np.save(os.path.join(OUT_DIR, f"mask_{idx}_{chunks_saved}.npy"), mask[y:y+IMG_SIZE, x:x+IMG_SIZE])

    return 1

# ============================================================
# 4. MAIN EXECUTION (Multiprocessing)
# ============================================================
if __name__ == "__main__":
    print(f"Starting Pre-processing with {NUM_WORKERS} CPU workers...")

    all_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpeg')])
    tasks = list(enumerate(all_files))

    total_chunks = 0
    # Use ProcessPoolExecutor to process images in parallel
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # tqdm gives us a nice progress bar
        for result in tqdm(executor.map(process_image, tasks), total=len(tasks), desc="Processing Images"):
            total_chunks += result

    print(f"\nSuccess! Generated {total_chunks} chunks in {OUT_DIR}")
    print("Next step: Zip the 'preprocessed_chunks' folder and upload it to Kaggle as a Dataset.")
