import os
import cv2
import json
import h5py
import numpy as np
from tqdm import tqdm

def load_pure_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Handle the common PURE vitals format
        return np.array([pt['Value'] for pt in data['vitals']])
    except Exception:
        return None

def preprocess():
    RAW_DATA_PATH = os.path.expanduser("~/DATASET_PURE")
    OUTPUT_FILE = os.path.expanduser("~/pure_processed_192.h5")
    IMG_SIZE = 192 

    # 1. Identify which subjects are already finished
    processed_subjects = []
    if os.path.exists(OUTPUT_FILE):
        with h5py.File(OUTPUT_FILE, 'r') as hf:
            processed_subjects = list(hf.keys())
            print(f"Found {len(processed_subjects)} subjects already processed. Resuming...")

    # 2. Get list of all subjects
    all_subjects = sorted([f for f in os.listdir(RAW_DATA_PATH) 
                          if os.path.isdir(os.path.join(RAW_DATA_PATH, f))])

    # 3. Open in 'a' (append) mode
    with h5py.File(OUTPUT_FILE, 'a') as hf:
        for sub in all_subjects:
            if sub in processed_subjects:
                continue # SKIP already done folders

            sub_path = os.path.join(RAW_DATA_PATH, sub)
            img_files = sorted([f for f in os.listdir(sub_path) if f.endswith('.png')])
            json_files = [f for f in os.listdir(sub_path) if f.endswith('.json')]
            
            if not json_files: continue
            bvp_signal = load_pure_data(os.path.join(sub_path, json_files[0]))
            if bvp_signal is None: continue
            
            frames = []
            num_to_process = min(len(img_files), len(bvp_signal))
            
            print(f"\n🚀 Processing {sub} ({num_to_process} frames)")
            for i in tqdm(range(num_to_process)):
                img = cv2.imread(os.path.join(sub_path, img_files[i]))
                if img is None: continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape
                # Center crop to 1:1
                side = min(h, w)
                img = img[(h-side)//2 : (h+side)//2, (w-side)//2 : (w+side)//2]
                frames.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))

            # Save this subject immediately to the file
            group = hf.create_group(sub)
            group.create_dataset('frames', data=np.array(frames), compression="gzip", compression_opts=4)
            group.create_dataset('bvp', data=bvp_signal[:num_to_process], compression="gzip")
            
            # Optional: Flush to disk after each subject
            hf.flush() 

if __name__ == "__main__":
    preprocess()