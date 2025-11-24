import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import csv
import cv2
from glob import glob

# --- CONFIGURATION ---
class Config:
    RAW_DATA_ROOT = "data"
    FINAL_OUTPUT_ROOT = "data/data_preprocessed"
    CSV_OUTPUT_DIR = "ACL/lists"
    TARGET_SIZE = (256, 256)
    CLIP_LIMIT = 2.0
    TILE_GRID_SIZE = (8, 8)

# --- UTILS FUNCTIONS ---

def ensure_directory_exists(path):
    """
    Ensures that the specified directory exists. Creates it if necessary.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Created directory: {path}")

def ensure_file_exists(file_path):
    """
    Ensures that the specified file exists. Creates an empty file if necessary.
    """
    if not os.path.exists(file_path):
        # Create directory for file if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            file.write("") 
        print(f"[INFO] Created file: {file_path}")

def get_orientation_subfolder(filename):
    """
    Determines the MRI orientation (axial, coronal, sagittal) based on the filename.
    Returns 'others' if unknown.
    """
    name_lower = filename.lower()
    if any(x in name_lower for x in ["axial", "tra", "axi"]):
        return "axial"
    elif any(x in name_lower for x in ["coronal", "cor"]):
        return "coronal"
    elif any(x in name_lower for x in ["sagittal", "sag"]):
        return "sagittal"
    else:
        return "others"

def center_crop_or_pad(image, target_size=(256, 256)):
    """
    Resizes the image to target_size by center cropping or padding with zeros.
    
    Args:
        image (np.ndarray): Input 2D image.
        target_size (tuple): Desired (height, width).
        
    Returns:
        np.ndarray: Processed image with shape target_size.
    """
    h, w = image.shape
    th, tw = target_size

    # 1. Padding (if image is smaller than target)
    pad_h = max(th - h, 0)
    pad_w = max(tw - w, 0)
    
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                       mode='constant', constant_values=0)

    # 2. Cropping (if image is larger than target)
    # Update dimensions after padding
    h, w = image.shape
    start_h = (h - th) // 2
    start_w = (w - tw) // 2
    
    return image[start_h:start_h+th, start_w:start_w+tw]

def process_image_slice(img_slice):
    """
    Applies preprocessing pipeline: Normalize -> CLAHE -> Normalize -> Crop/Pad.
    SAFE method: Handles float conversion carefully to avoid overflow.
    """
    # 1. Convert to float32 first to safely handle high intensity values
    img_float = img_slice.astype(np.float32)

    # 2. Robust Min-Max Normalization (1st Pass)
    # Using percentiles to remove outliers
    p1 = np.percentile(img_float, 1)
    p99 = np.percentile(img_float, 99)
    
    if p99 - p1 == 0:
        norm_img = np.zeros_like(img_float)
    else:
        norm_img = (img_float - p1) / (p99 - p1)
    
    norm_img = np.clip(norm_img, 0, 1)

    # 3. Apply CLAHE (Requires uint8 [0-255])
    img_uint8 = (norm_img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=Config.CLIP_LIMIT, tileGridSize=Config.TILE_GRID_SIZE)
    img_clahe = clahe.apply(img_uint8)

    # 4. Convert back to Float [0, 1] for Model Training
    final_img = img_clahe.astype(np.float32) / 255.0

    # 5. Spatial Transform (Crop/Pad)
    final_img = center_crop_or_pad(final_img, target_size=Config.TARGET_SIZE)
    
    return final_img

def generate_file_pairs(root_dir):
    """
    Scans the directory to match Image and Mask files based on orientation.
    """
    file_pairs = []
    if not os.path.exists(root_dir):
        print(f"[ERROR] Directory '{root_dir}' not found.")
        return []

    print(f"[INFO] Scanning directory: {root_dir}...")
    
    case_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')])

    for case_id in case_dirs:
        case_path = os.path.join(root_dir, case_id)
        image_files = {} 
        mask_files = {} 
        
        for filename in os.listdir(case_path):
            if filename.endswith(".nrrd"):
                orientation = get_orientation_subfolder(filename)
                # Check if it's a mask or an image
                if ".seg." in filename.lower() or "mask" in filename.lower():
                    mask_files[orientation] = filename
                else:
                    image_files[orientation] = filename

        # Pair them up
        for view, img_name in image_files.items():
            mask_name = mask_files.get(view)
            if mask_name:
                unique_id = f"{case_id}_{view}"
                file_pairs.append({
                    "case_id": case_id,
                    "view": view,
                    "image": img_name,
                    "mask": mask_name,
                    "id": unique_id
                })
                
    print(f"[SUCCESS] Found {len(file_pairs)} paired volumes.")
    return file_pairs

# --- MAIN EXECUTION ---
def main():
    ensure_directory_exists(Config.FINAL_OUTPUT_ROOT)
    file_pairs = generate_file_pairs(Config.RAW_DATA_ROOT)

    if not file_pairs:
        print("[WARN] No data found. Exiting...")
        return

    # 1. Processing Volumes and Masks
    print("\n[INFO] Starting Preprocessing Pipeline...")
    
    for idx, pair in enumerate(file_pairs):
        case_id = pair["case_id"]
        view = pair["view"]
        unique_id = pair["id"]
        
        print(f"[{idx+1}/{len(file_pairs)}] Processing {unique_id}...", end='\r')

        # Paths
        vol_path = os.path.join(Config.RAW_DATA_ROOT, case_id, pair["image"])
        mask_path = os.path.join(Config.RAW_DATA_ROOT, case_id, pair["mask"])

        # Create output directories
        img_save_dir = os.path.join(Config.FINAL_OUTPUT_ROOT, case_id, f"{view}_img")
        mask_save_dir = os.path.join(Config.FINAL_OUTPUT_ROOT, case_id, f"{view}_mask")
        ensure_directory_exists(img_save_dir)
        ensure_directory_exists(mask_save_dir)

        # Read Files
        try:
            vol_itk = sitk.ReadImage(vol_path)
            mask_itk = sitk.ReadImage(mask_path)
            
            vol_arr = sitk.GetArrayFromImage(vol_itk)
            mask_arr = sitk.GetArrayFromImage(mask_itk)
        except Exception as e:
            print(f"\n[ERROR] Failed to read {unique_id}: {e}")
            continue

        # Iterate through slices
        for i in range(vol_arr.shape[0]):
            # -- Process Image --
            processed_img = process_image_slice(vol_arr[i, :, :])
            
            # -- Process Mask --
            # Mask only needs Crop/Pad (No normalization)
            mask_slice = mask_arr[i, :, :]
            processed_mask = center_crop_or_pad(mask_slice, target_size=Config.TARGET_SIZE)

            # Save Files
            np.save(os.path.join(img_save_dir, f"{unique_id}_vol_slice_{i}.npy"), processed_img)
            np.save(os.path.join(mask_save_dir, f"{unique_id}_mask_slice_{i}.npy"), processed_mask)

    print(f"\n[SUCCESS] Preprocessing completed.")

    # 2. Generating CSV and Lists
    print("\n[INFO] Generating training lists...")
    ensure_directory_exists(Config.CSV_OUTPUT_DIR)
    csv_path = os.path.join(Config.CSV_OUTPUT_DIR, "train.csv")

    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['image', 'mask'])
        
        for pair in file_pairs:
            case_id = pair["case_id"]
            view = pair["view"]
            unique_id = pair["id"]
            
            # Use mask depth to determine number of slices
            mask_path = os.path.join(Config.RAW_DATA_ROOT, case_id, pair["mask"])
            try:
                # Use simpleITK to check depth quickly without loading full array again
                reader = sitk.ImageFileReader()
                reader.SetFileName(mask_path)
                reader.ReadImageInformation()
                num_slices = reader.GetSize()[2] # Dimension 2 is depth (z) in ITK convention (x,y,z)
                # Note: sitk.GetArrayFromImage swaps to (z,y,x), but ReadImageInformation keeps (x,y,z)
            except:
                continue

            for i in range(num_slices):
                # Relative paths for CSV
                vol_rel_path = f"{case_id}/{view}_img/{unique_id}_vol_slice_{i}.npy"
                seg_rel_path = f"{case_id}/{view}_mask/{unique_id}_mask_slice_{i}.npy"
                writer.writerow([vol_rel_path, seg_rel_path])

    # 3. Generating TXT Lists
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Helper to write list
        def write_list(series, filename):
            path = os.path.join(Config.CSV_OUTPUT_DIR, filename)
            # Extract filename without extension for list
            names = [os.path.splitext(os.path.basename(p))[0] for p in series]
            with open(path, 'w') as f:
                f.write('\n'.join(names))
            print(f"[INFO] Saved list to: {path}")

        write_list(df['image'], "train_image.txt")
        write_list(df['mask'], "train_mask.txt")

    print("[SUCCESS] All tasks finished successfully.")

if __name__ == "__main__":
    main()