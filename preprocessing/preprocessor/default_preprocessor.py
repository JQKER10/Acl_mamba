import os
import pickle
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import multiprocessing
import tqdm
from time import sleep

class TwoStagePreprocessor2_5D:
    """
    Stage 1: Volume-level preprocessing cho 2.5D segmentation
    
    Chức năng:
        - Load volume từ .nii/.nii.gz/.mhd
        - Normalize intensity (z-score, min-max, CT window)
        - Resample in-plane về target spacing
        - Crop (global bbox hoặc per-slice)
        - Lưu thành .npy format cho fast loading
    
    Output (nnU-Net style):
        - {case_id}.npy       : image (C, D, H, W), C=1
        - {case_id}_seg.npy   : label (D, H, W)
        - {case_id}.pkl       : properties dict
    
    Note: num_neighbors KHÔNG CẦN ở Stage 1
          Stage 2 (trong Dataset) mới cần neighbors để tạo 2.5D samples
    """
    
    def __init__(
        self,
        target_spacing_xy: Tuple[float, float] = (1.0, 1.0),
        intensity_normalization: str = "global_zscore",
        use_global_crop: bool = False,
        crop_margin: int = 10,
    ):
        """
        Args:
            target_spacing_xy: Target in-plane spacing (Y, X) in mm
            intensity_normalization: 
                - 'global_zscore': Z-score normalization (mean=0, std=1)
                - 'global_minmax': Min-max to [0,1] với percentile clipping
                - 'ct_window': CT soft-tissue window [-125, 275] HU
                - 'none': No normalization
            use_global_crop: 
                - True: Global bbox cho toàn volume (consistent shape)
                - False: Per-slice crop (optimize memory, variable shape)
            crop_margin: Margin pixels khi crop
        """
        self.target_spacing_xy = target_spacing_xy
        self.intensity_normalization = intensity_normalization
        self.use_global_crop = use_global_crop
        self.crop_margin = crop_margin
        
        print("=" * 70)
        print("TwoStagePreprocessor2_5D (Stage 1: Volume Preprocessing)")
        print("=" * 70)
        print(f"Target spacing (Y, X): {target_spacing_xy} mm")
        print(f"Normalization method: {intensity_normalization}")
        print(f"Crop strategy: {'Global bbox' if use_global_crop else 'Per-slice'}")
        print(f"Crop margin: {crop_margin} pixels")
        print("=" * 70)
    
    # ==================== STATISTICS & NORMALIZATION ====================
    
    def compute_global_statistics(
        self, 
        volume: np.ndarray, 
        seg: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Tính global statistics từ volume
        
        Args:
            volume: Image array (D, H, W)
            seg: Optional segmentation mask (D, H, W)
        
        Returns:
            Dictionary chứa mean, std, percentiles, etc.
        """
        # Chỉ tính trên foreground nếu có seg
        if seg is not None and seg.sum() > 0:
            foreground = volume[seg > 0]
        else:
            # Foreground = pixels > background
            foreground = volume[volume > volume.min()]
        
        # Fallback nếu không có foreground
        if len(foreground) == 0:
            foreground = volume.flatten()
        
        stats = {
            'mean': float(np.mean(foreground)),
            'std': float(np.std(foreground)),
            'median': float(np.median(foreground)),
            'min': float(np.min(foreground)),
            'max': float(np.max(foreground)),
            'p01': float(np.percentile(foreground, 1)),
            'p99': float(np.percentile(foreground, 99)),
            'p005': float(np.percentile(foreground, 0.5)),
            'p995': float(np.percentile(foreground, 99.5)),
        }
        
        return stats
    
    def normalize_volume_global(
        self, 
        volume: np.ndarray, 
        stats: Dict
    ) -> np.ndarray:
        """
        Normalize toàn volume theo global statistics
        
        Args:
            volume: Image array (D, H, W)
            stats: Statistics dict từ compute_global_statistics()
        
        Returns:
            Normalized volume
        """
        volume = volume.astype(np.float32)
        
        if self.intensity_normalization == "global_zscore":
            # Z-score: (x - mean) / std
            normalized = (volume - stats["mean"]) / (stats["std"] + 1e-8)
            # CRITICAL: Clip outliers để tránh extreme values
            normalized = np.clip(normalized, -5, 5)
            return normalized
        
        elif self.intensity_normalization == "global_minmax":
            # Percentile-based min-max normalization
            p_low, p_high = stats["p005"], stats["p995"]
            clipped = np.clip(volume, p_low, p_high)
            normalized = (clipped - p_low) / (p_high - p_low + 1e-8)
            return normalized
        
        elif self.intensity_normalization == "ct_window":
            # CT soft-tissue window: [-125, 275] HU → [0, 1]
            clipped = np.clip(volume, -125, 275)
            normalized = (clipped + 125) / 400.0
            return normalized
        
        elif self.intensity_normalization == "none":
            return volume
        
        else:
            raise ValueError(f"Unknown normalization: {self.intensity_normalization}")
    
    # ==================== RESAMPLING ====================
    
    def resample_slice_inplane(
        self, 
        slice_2d: np.ndarray, 
        current_spacing: Tuple[float, float], 
        is_label: bool = False
    ) -> np.ndarray:
        """
        Resample 1 slice 2D trong mặt phẳng (Y, X)
        
        Args:
            slice_2d: 2D array (H, W)
            current_spacing: Current spacing (Y, X) in mm
            is_label: True nếu là segmentation mask
        
        Returns:
            Resampled 2D array
        """
        zoom_y = current_spacing[0] / self.target_spacing_xy[0]
        zoom_x = current_spacing[1] / self.target_spacing_xy[1]
        
        # Skip nếu không cần resample
        if abs(zoom_y - 1.0) < 0.01 and abs(zoom_x - 1.0) < 0.01:
            return slice_2d
        
        # Order 0 (nearest) cho label, order 3 (cubic) cho image
        order = 0 if is_label else 3
        resampled = zoom(slice_2d, (zoom_y, zoom_x), order=order)
        
        return resampled
    
    # ==================== CROPPING ====================
    
    def crop_to_nonzero_2d(
        self, 
        slice_2d: np.ndarray, 
        margin: Optional[int] = None
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Crop 1 slice 2D theo non-zero region
        
        Args:
            slice_2d: 2D array (H, W)
            margin: Margin pixels (default: self.crop_margin)
        
        Returns:
            cropped_array: Cropped 2D array
            bbox: (y_min, y_max, x_min, x_max)
        """
        if margin is None:
            margin = self.crop_margin
        
        # Find non-zero region
        mask = slice_2d > slice_2d.min()
        
        if not mask.any():
            # No foreground, return original
            return slice_2d, (0, slice_2d.shape[0], 0, slice_2d.shape[1])
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]
        
        y_min, y_max = y_indices[0], y_indices[-1]
        x_min, x_max = x_indices[0], x_indices[-1]
        
        # Add margin
        H, W = slice_2d.shape
        y_min = max(0, y_min - margin)
        y_max = min(H, y_max + margin + 1)
        x_min = max(0, x_min - margin)
        x_max = min(W, x_max + margin + 1)
        
        cropped = slice_2d[y_min:y_max, x_min:x_max]
        bbox = (y_min, y_max, x_min, x_max)
        
        return cropped, bbox
    
    def compute_global_bbox_3d(
        self, 
        volume: np.ndarray, 
        margin: Optional[int] = None
    ) -> Tuple[int, int, int, int, int, int]:
        """
        Compute global bounding box cho toàn volume
        
        Args:
            volume: 3D array (D, H, W)
            margin: Margin pixels (default: self.crop_margin)
        
        Returns:
            bbox: (z_min, z_max, y_min, y_max, x_min, x_max)
        """
        if margin is None:
            margin = self.crop_margin
        
        mask = volume > volume.min()
        
        if not mask.any():
            D, H, W = volume.shape
            return (0, D, 0, H, 0, W)
        
        # Find non-zero in each dimension
        z_any = mask.any(axis=(1, 2))
        y_any = mask.any(axis=(0, 2))
        x_any = mask.any(axis=(0, 1))
        
        z_indices = np.where(z_any)[0]
        y_indices = np.where(y_any)[0]
        x_indices = np.where(x_any)[0]
        
        if len(z_indices) == 0 or len(y_indices) == 0 or len(x_indices) == 0:
            D, H, W = volume.shape
            return (0, D, 0, H, 0, W)
        
        z_min, z_max = z_indices[0], z_indices[-1]
        y_min, y_max = y_indices[0], y_indices[-1]
        x_min, x_max = x_indices[0], x_indices[-1]
        
        # Add margin
        D, H, W = volume.shape
        z_min = max(0, z_min - margin)
        z_max = min(D, z_max + margin + 1)
        y_min = max(0, y_min - margin)
        y_max = min(H, y_max + margin + 1)
        x_min = max(0, x_min - margin)
        x_max = min(W, x_max + margin + 1)
        
        return (z_min, z_max, y_min, y_max, x_min, x_max)
    
    # ==================== MAIN PREPROCESSING ====================
    
    def preprocess_single_volume(
        self, 
        case_id: str, 
        image_path: str, 
        label_path: Optional[str] = None
    ) -> Dict:
        """
        Preprocess 1 volume (Stage 1 complete pipeline)
        
        Args:
            case_id: Case identifier (e.g., "case_000")
            image_path: Path to image file (.nii.gz, .mhd, etc.)
            label_path: Optional path to segmentation file
        
        Returns:
            Dictionary với keys:
                - volume: Processed image array
                - seg: Processed segmentation array (or None)
                - properties: Metadata dict
        """
        # ---- 1. Load image ----
        img_sitk = sitk.ReadImage(image_path)
        img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)  # (D, H, W)
        spacing_zyx = img_sitk.GetSpacing()[::-1]  # (Z, Y, X)
        
        # ---- 2. Load segmentation (optional) ----
        seg_arr = None
        if label_path and os.path.exists(label_path):
            seg_sitk = sitk.ReadImage(label_path)
            seg_arr = sitk.GetArrayFromImage(seg_sitk).astype(np.uint8)
            
            # Validate shape
            if img_arr.shape != seg_arr.shape:
                raise ValueError(
                    f"Shape mismatch for {case_id}: "
                    f"image {img_arr.shape} vs seg {seg_arr.shape}"
                )
        
        D, H, W = img_arr.shape
        
        # ---- 3. Compute global statistics ----
        stats = self.compute_global_statistics(img_arr, seg_arr)
        
        # ---- 4. Normalize volume ----
        normalized_volume = self.normalize_volume_global(img_arr, stats)
        
        # ---- 5. Crop + Resample ----
        if self.use_global_crop:
            # Strategy A: Global bbox (consistent shape across slices)
            processed_volume, processed_seg, bboxes = self._process_with_global_crop(
                normalized_volume, seg_arr, spacing_zyx
            )
        else:
            # Strategy B: Per-slice crop (optimize memory, variable shapes)
            processed_volume, processed_seg, bboxes = self._process_with_perslice_crop(
                normalized_volume, seg_arr, spacing_zyx
            )
        
        # ---- 6. Build properties dict ----
        properties = {
            "case_id": case_id,
            "original_shape": (D, H, W),
            "original_spacing": spacing_zyx,
            "processed_shape": processed_volume.shape,
            "processed_spacing": (spacing_zyx[0], *self.target_spacing_xy),
            "bboxes": bboxes,  # List hoặc single bbox
            "statistics": stats,
            "normalization": {
                "method": self.intensity_normalization,
                "parameters": stats,
            },
            "preprocessing_config": {
                "target_spacing_xy": self.target_spacing_xy,
                "use_global_crop": self.use_global_crop,
                "crop_margin": self.crop_margin,
            }
        }
        
        return {
            "volume": processed_volume,
            "seg": processed_seg,
            "properties": properties,
        }
    
    def _process_with_global_crop(
        self, 
        volume: np.ndarray, 
        seg: Optional[np.ndarray], 
        spacing_zyx: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List]:
        """
        Process với global bbox (consistent shape)
        """
        # Compute global bbox
        bbox_3d = self.compute_global_bbox_3d(volume)
        z_min, z_max, y_min, y_max, x_min, x_max = bbox_3d
        
        # Crop volume
        cropped_vol = volume[z_min:z_max, y_min:y_max, x_min:x_max]
        cropped_seg = seg[z_min:z_max, y_min:y_max, x_min:x_max] if seg is not None else None
        
        D_crop = cropped_vol.shape[0]
        
        # Estimate output shape bằng cách resample 1 slice giữa
        sample_slice = self.resample_slice_inplane(
            cropped_vol[D_crop // 2],
            current_spacing=(spacing_zyx[1], spacing_zyx[2]),
            is_label=False
        )
        target_H, target_W = sample_slice.shape
        
        # Preallocate arrays
        processed_volume = np.zeros((D_crop, target_H, target_W), dtype=np.float32)
        processed_seg = np.zeros((D_crop, target_H, target_W), dtype=np.uint8) if cropped_seg is not None else None
        
        # Resample mỗi slice
        for z in range(D_crop):
            processed_volume[z] = self.resample_slice_inplane(
                cropped_vol[z],
                current_spacing=(spacing_zyx[1], spacing_zyx[2]),
                is_label=False
            )
            
            if processed_seg is not None:
                processed_seg[z] = self.resample_slice_inplane(
                    cropped_seg[z],
                    current_spacing=(spacing_zyx[1], spacing_zyx[2]),
                    is_label=True
                )
        
        # All slices have same bbox
        bboxes = [bbox_3d] * D_crop
        
        return processed_volume, processed_seg, bboxes
    
    def _process_with_perslice_crop(
        self, 
        volume: np.ndarray, 
        seg: Optional[np.ndarray], 
        spacing_zyx: Tuple[float, float, float]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List]:
        """
        Process với per-slice crop (variable shapes)
        """
        D = volume.shape[0]
        
        processed_slices: List[np.ndarray] = []
        processed_seg_slices: Optional[List[np.ndarray]] = [] if seg is not None else None
        bboxes: List[Tuple[int, int, int, int]] = []
        
        for z in range(D):
            slice_img = volume[z]
            
            # Crop per-slice
            cropped_img, bbox = self.crop_to_nonzero_2d(slice_img)
            bboxes.append(bbox)
            
            # Resample in-plane
            resampled_img = self.resample_slice_inplane(
                cropped_img,
                current_spacing=(spacing_zyx[1], spacing_zyx[2]),
                is_label=False,
            )
            processed_slices.append(resampled_img)
            
            # Process seg
            if seg is not None:
                y_min, y_max, x_min, x_max = bbox
                cropped_seg = seg[z, y_min:y_max, x_min:x_max]
                
                resampled_seg = self.resample_slice_inplane(
                    cropped_seg,
                    current_spacing=(spacing_zyx[1], spacing_zyx[2]),
                    is_label=True,
                )
                processed_seg_slices.append(resampled_seg)
        
        # Stack slices (shapes có thể khác nhau - will handle in collate_fn)
        processed_volume = np.stack(processed_slices, axis=0)
        processed_seg = np.stack(processed_seg_slices, axis=0) if processed_seg_slices else None
        
        return processed_volume, processed_seg, bboxes
    
    # ==================== SAVE RESULTS ====================
    
    def save_stage1_result(self, result: Dict, output_dir: str):
        """
        Lưu kết quả Stage 1 theo nnU-Net format
        
        Args:
            result: Dict từ preprocess_single_volume()
            output_dir: Output directory path
        
        Output files:
            - {case_id}.npy: Image (1, D, H, W)
            - {case_id}_seg.npy: Segmentation (D, H, W)
            - {case_id}.pkl: Properties dict
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        case_id = result["properties"]["case_id"]
        vol = result["volume"]  # (D, H, W)
        seg = result["seg"]     # (D, H, W) or None
        props = result["properties"]
        
        # Save image: add channel dimension (1, D, H, W)
        img_npy = vol[None, ...].astype(np.float32)
        np.save(output_dir / f"{case_id}.npy", img_npy)
        
        # Save segmentation: (D, H, W)
        if seg is not None:
            seg_npy = seg.astype(np.int16)
            np.save(output_dir / f"{case_id}_seg.npy", seg_npy)
        
        # Save properties
        with open(output_dir / f"{case_id}.pkl", "wb") as f:
            pickle.dump(props, f)
    
    # ==================== MULTIPROCESSING ====================
    
    def _process_and_save_stage1_worker(self, args):
        """
        Worker function cho multiprocessing
        
        Args:
            args: Tuple of (case_id, image_path, label_path, output_dir)
        
        Returns:
            (success: bool, message: str)
        """
        case_id, image_path, label_path, output_dir = args
        
        try:
            result = self.preprocess_single_volume(case_id, image_path, label_path)
            self.save_stage1_result(result, output_dir)
            return True, case_id
        except Exception as e:
            import traceback
            error_msg = f"{case_id}: {str(e)}\n{traceback.format_exc()}"
            return False, error_msg
    
    def run_stage1(
        self, 
        image_dir: str, 
        output_dir: str, 
        label_dir: Optional[str] = None, 
        case_ids: Optional[List[str]] = None, 
        num_processes: int = 8, 
        file_extension: str = ".nii.gz"
    ):
        """
        Run Stage 1 preprocessing cho toàn bộ dataset
        
        Args:
            image_dir: Directory chứa image files
            output_dir: Output directory
            label_dir: Optional directory chứa segmentation files
            case_ids: Optional list of case IDs. Nếu None, auto-detect từ image_dir
            num_processes: Number of parallel processes
            file_extension: File extension (.nii.gz, .nii, .mhd, etc.)
        """
        from multiprocessing import Pool
        from tqdm import tqdm
        
        print("\n" + "=" * 70)
        print("STAGE 1: VOLUME-LEVEL PREPROCESSING")
        print("=" * 70)
        
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ---- Get case list ----
        if case_ids is None:
            case_ids = []
            for f in image_dir.glob(f"*{file_extension}"):
                case_id = f.name[:-len(file_extension)]
                case_ids.append(case_id)
            case_ids = sorted(set(case_ids))
        
        print(f"\nDataset info:")
        print(f"  Input directory:  {image_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Label directory:  {label_dir if label_dir else 'None (test mode)'}")
        print(f"  File extension:   {file_extension}")
        print(f"  Found cases:      {len(case_ids)}")
        print(f"  Num processes:    {num_processes}")
        
        # ---- Prepare arguments ----
        args_list = []
        for case_id in case_ids:
            image_path = str(image_dir / f"{case_id}{file_extension}")
            label_path = str(Path(label_dir) / f"{case_id}{file_extension}") if label_dir else None
            args_list.append((case_id, image_path, label_path, str(output_dir)))
        
        if len(args_list) == 0:
            print("\n✗ No cases found!")
            return
        
        # ---- Test run with first case ----
        print("\n" + "-" * 70)
        print("Test run with first case...")
        print("-" * 70)
        success, msg = self._process_and_save_stage1_worker(args_list[0])
        
        if not success:
            print(f"\n✗ Test failed!")
            print(f"Error: {msg}")
            return
        
        print(f"✓ Test successful: {msg}")
        
        if len(args_list) == 1:
            print("\nOnly 1 case, preprocessing completed.")
            self._print_summary(1, 0)
            return
        
        # ---- Process remaining cases with multiprocessing ----
        print(f"\n" + "-" * 70)
        print(f"Processing remaining {len(args_list) - 1} cases...")
        print("-" * 70)
        
        results = []
        with Pool(num_processes) as pool:
            for success, msg in tqdm(
                pool.imap(self._process_and_save_stage1_worker, args_list[1:]),
                total=len(args_list) - 1,
                desc="Preprocessing",
                unit="case"
            ):
                results.append((success, msg))
                if not success:
                    print(f"\n✗ Failed: {msg}")
        
        # ---- Summary ----
        success_count = sum(1 for s, _ in results if s) + 1  # +1 for test case
        failed_count = len(args_list) - success_count
        
        self._print_summary(success_count, failed_count)
        
        # Print failed cases
        if failed_count > 0:
            print("\nFailed cases:")
            for success, msg in results:
                if not success:
                    print(f"  - {msg.split(':')[0]}")
    
    def _print_summary(self, success_count: int, failed_count: int):
        """Print preprocessing summary"""
        total = success_count + failed_count
        
        print("\n" + "=" * 70)
        print("STAGE 1 COMPLETED")
        print("=" * 70)
        print(f"✓ Success: {success_count}/{total} ({success_count/total*100:.1f}%)")
        if failed_count > 0:
            print(f"✗ Failed:  {failed_count}/{total} ({failed_count/total*100:.1f}%)")
        print("=" * 70)
