import os
import pickle
import random
from typing import Optional, Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class Seg2p5DMedicalDataset(Dataset):
    def __init__(
        self,
        datalist: List[str],
        neighbor_slices: int = 4,
        test: bool = False,
        use_properties: bool = True,
        cache_volumes: bool = False,
        preprocess_fn: Optional[callable] = None,
        padding_mode: str = "reflect",
        oversample_foreground_percent: float = 0.5 
    ) -> None:
        super().__init__()

        assert neighbor_slices % 2 == 0, "neighbor_slices phải chẵn"
        assert 0.0 <= oversample_foreground_percent <= 1.0, "Percent phải từ 0 đến 1"

        self.datalist = datalist
        self.test = test
        self.use_properties = use_properties
        self.cache_volumes = cache_volumes
        self.preprocess_fn = preprocess_fn
        self.K = neighbor_slices
        self.half = self.K // 2
        self.padding_mode = padding_mode
        self.oversample_foreground_percent = oversample_foreground_percent

        self._validate_files()

        # Load metadata
        self.properties_cached = []
        if self.use_properties:
            print("Loading properties (.pkl)...")
            for p in tqdm(self.datalist, desc="Loading metadata"):
                self.properties_cached.append(self._load_pkl(p))
        else:
            self.properties_cached = [None] * len(self.datalist)

        # --- INDEXING ---
        print("Indexing volumes (checking FG/BG)...")
        self.fg_samples: List[Tuple[int, int]] = [] 
        self.bg_samples: List[Tuple[int, int]] = [] 
        self.all_samples: List[Tuple[int, int]] = [] 

        for case_idx, p in enumerate(tqdm(self.datalist, desc="Scanning volumes")):
            # [UPDATE - Mục 4] Xử lý path an toàn hơn
            base, _ = os.path.splitext(p)
            image_path = base + ".npy"
            
            img_mmap = np.load(image_path, mmap_mode="r")
            
            if img_mmap.ndim == 3:
                D, H, W = img_mmap.shape
            else:
                C, D, H, W = img_mmap.shape

            # Load seg mmap
            seg_mmap = None
            if not self.test:
                seg_path = base + "_seg.npy" # [UPDATE - Mục 4]
                if os.path.exists(seg_path):
                    seg_mmap = np.load(seg_path, mmap_mode="r")
                    if seg_mmap.ndim == 4: seg_mmap = seg_mmap[0]

            for z in range(D):
                sample_tuple = (case_idx, z)
                self.all_samples.append(sample_tuple)

                is_fg = False
                if seg_mmap is not None:
                    if np.any(seg_mmap[z] > 0):
                        is_fg = True
                
                if is_fg:
                    self.fg_samples.append(sample_tuple)
                else:
                    self.bg_samples.append(sample_tuple)

        # --- BUILD FINAL SAMPLE LIST (Mục 3) ---
        if self.test or self.oversample_foreground_percent == 0.0 or len(self.fg_samples) == 0:
            self.samples = self.all_samples
            print(f"Oversampling OFF or Test mode. Total samples: {len(self.samples)}")
        else:
            print(f"Applying Oversampling: Target {self.oversample_foreground_percent*100}% FG")
            self.samples = self._build_oversampled_list()
        
        # Cache logic
        self.volume_cache = {}
        if self.cache_volumes:
            print("Caching full volumes to RAM...")
            for case_idx, p in enumerate(tqdm(self.datalist, desc="Caching volumes")):
                img, seg = self._read_full_volume(p)
                self.volume_cache[case_idx] = (img, seg)

        print(f"Dataset ready. Final dataset length: {len(self.samples)}")

    def _build_oversampled_list(self) -> List[Tuple[int, int]]:
        """
        [UPDATE - Mục 3] Cải thiện logic Oversampling
        """
        n_bg = len(self.bg_samples)
        n_fg = len(self.fg_samples)
        target = self.oversample_foreground_percent

        if target >= 1.0: return self.fg_samples

        # Tính số lượng FG mới cần thiết để đạt tỷ lệ target
        # n_new_fg / (n_new_fg + n_bg) = target
        n_new_fg = int(n_bg * target / (1 - target))
        n_new_fg = max(n_new_fg, n_fg) # Không bao giờ lấy ít hơn số FG gốc

        # Random sampling with replacement
        new_fg_samples = random.choices(self.fg_samples, k=n_new_fg)
        
        final_list = new_fg_samples + self.bg_samples
        random.shuffle(final_list)
        
        print(f"   Stats: Original FG: {n_fg}, BG: {n_bg}")
        print(f"   Stats: Resampled FG: {len(new_fg_samples)} + BG: {n_bg} = Total: {len(final_list)}")
        return final_list

    def _validate_files(self):
        for data_path in self.datalist:
            # [UPDATE - Mục 4] Path an toàn
            base, _ = os.path.splitext(data_path)
            
            image_path = base + ".npy"
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            if self.use_properties:
                pkl_path = base + ".pkl"
                if not os.path.exists(pkl_path):
                    raise FileNotFoundError(f"Properties file not found: {pkl_path}")
            
            if not self.test:
                seg_path = base + "_seg.npy"
                if not os.path.exists(seg_path):
                    raise FileNotFoundError(f"Segmentation file not found: {seg_path}")

    def _load_pkl(self, data_path: str) -> Dict:
        base, _ = os.path.splitext(data_path)
        properties_path = base + ".pkl"
        with open(properties_path, "rb") as f:
            props = pickle.load(f)
        return props

    def _read_full_volume(self, data_path: str):
        base, _ = os.path.splitext(data_path)
        img = np.load(base + ".npy")
        seg = None
        if not self.test:
            seg = np.load(base + "_seg.npy")
        
        if img.ndim == 3: img = img[None, ...]
        if (seg is not None) and (seg.ndim == 4): seg = seg[0]
        return img, seg

    def _read_volume_mmap(self, data_path: str):
        base, _ = os.path.splitext(data_path)
        img = np.load(base + ".npy", mmap_mode="r")
        seg = None
        if not self.test:
            seg = np.load(base + "_seg.npy", mmap_mode="r")
            
        if img.ndim == 3: img = img[None, ...]
        if (seg is not None) and (seg.ndim == 4): seg = seg[0]
        return img, seg

    def _get_slice(self, vol: np.ndarray, z: int):
        if vol.ndim == 4: return vol[:, z, :, :]
        else: return vol[z, :, :]

    def _get_neighbor_index(self, z: int, offset: int, D: int) -> int:
        z_n = z + offset
        if self.padding_mode == "reflect":
            if z_n < 0: z_n = -z_n
            elif z_n >= D: z_n = 2 * D - z_n - 2
            z_n = np.clip(z_n, 0, D - 1)
        elif self.padding_mode == "replicate":
            z_n = np.clip(z_n, 0, D - 1)
        elif self.padding_mode == "zero":
            pass 
        return z_n

    def _get_slice_safe(self, vol: np.ndarray, z: int, D: int):
        # Trả về slice đen (zeros) nếu ra ngoài biên (cho chế độ padding zero)
        if z < 0 or z >= D:
            if vol.ndim == 4:
                C, _, H, W = vol.shape
                return np.zeros((C, H, W), dtype=vol.dtype)
            else:
                _, H, W = vol.shape
                return np.zeros((H, W), dtype=vol.dtype)
        else:
            return self._get_slice(vol, z)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        # [UPDATE - Mục 2] Tối ưu hóa toàn bộ hàm này
        case_idx, z = self.samples[idx]
        data_path = self.datalist[case_idx]

        # 1. Load Volume (Cache hoặc MMap)
        if self.cache_volumes:
            img_vol, seg_vol = self.volume_cache[case_idx]
        else:
            img_vol, seg_vol = self._read_volume_mmap(data_path)

        if img_vol.ndim == 4: C, D, H, W = img_vol.shape
        else:
            D, H, W = img_vol.shape
            C = 1

        # 2. Get 2.5D Stack (Center + Neighbors)
        # Thay vì tách center riêng, ta gộp chung để tạo 1 Tensor input duy nhất
        slice_stack = []
        
        for offset in range(-self.half, self.half + 1):
            # offset = 0 chính là center slice
            if self.padding_mode == "zero":
                z_n = z + offset
                slice_v = self._get_slice_safe(img_vol, z_n, D)
            else:
                z_n = self._get_neighbor_index(z, offset, D)
                slice_v = self._get_slice(img_vol, z_n)
            
            slice_stack.append(slice_v)

        # 3. Stack & Convert to Tensor
        # Kết quả: shape (K, H, W) hoặc (K, C, H, W)
        input_np = np.stack(slice_stack, axis=0) 

        # Xử lý shape để phù hợp với Conv2d/3d
        # Nếu ảnh gốc có C channels: (K, C, H, W) -> Muốn thành (C*K, H, W) hoặc giữ nguyên tùy model
        # Ở đây ta giả sử model nhận input (Batch, Channel, Height, Width)
        # Nên ta flatten dimension K và C
        if input_np.ndim == 4: # (K, C, H, W)
            # Transpose về (C, K, H, W) rồi reshape
            input_np = input_np.transpose(1, 0, 2, 3) # -> (C, K, H, W)
            input_np = input_np.reshape(-1, H, W)     # -> (C*K, H, W)
        
        # [Memory Fix] Không dùng .copy(), dùng trực tiếp
        input_tensor = torch.from_numpy(input_np).float()
        
        # 4. Get Mask (chỉ cần Center slice)
        mask_tensor = None
        if seg_vol is not None:
            mask_np = self._get_slice(seg_vol, z)
            mask_tensor = torch.from_numpy(mask_np).long()

        properties = self.properties_cached[case_idx] if self.use_properties else None

        # 5. Preprocessing (nếu có)
        # Lưu ý: preprocess_fn của bạn cần hỗ trợ input shape mới này
        if self.preprocess_fn is not None:
            # input_tensor chứa cả center và neighbors
            input_tensor, mask_tensor = self.preprocess_fn(input_tensor, mask_tensor)

        return {
            "image": input_tensor,      # Input chính cho model (đã gộp neighbors)
            "mask": mask_tensor,
            "properties": properties,
            "case_idx": case_idx,
            "z_idx": z,
            "global_idx": idx,
        }