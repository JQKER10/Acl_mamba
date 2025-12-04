import os
import pickle
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Seg2p5DMedicalDataset(Dataset):
    """
    Dataset 2.5D với PADDING để mọi slice đều có đủ neighbors:
    - Đầu vào từ NNUnet-style: .npy, _seg.npy, .pkl
    - Trả ra từng sample theo lát z: center + neighbors + mask center.
    - Xử lý edge cases bằng reflection/replication padding
    
    Giả định:
    - image.npy: (C, D, H, W)
    - seg.npy  : (D, H, W)  (nếu (1, D, H, W) thì tự squeeze trong code)
    """

    def __init__(
        self,
        datalist: List[str],      # list path .npz (prefix), ví dụ ".../case_000.npz"
        neighbor_slices: int = 4, # K = tổng số neighbor (2 trước + 2 sau)
        test: bool = False,
        use_properties: bool = True,
        cache_volumes: bool = False,   # cache cả volume vào RAM (nếu dataset nhỏ)
        preprocess_fn: Optional[callable] = None,  # augment/normalize, v.v.
        padding_mode: str = "reflect",  # 'reflect', 'replicate', 'zero'
    ) -> None:
        super().__init__()

        assert neighbor_slices % 2 == 0, "neighbor_slices phải chẵn (2 trước + 2 sau, v.v.)"

        self.datalist = datalist
        self.test = test
        self.use_properties = use_properties
        self.cache_volumes = cache_volumes
        self.preprocess_fn = preprocess_fn
        self.K = neighbor_slices
        self.half = self.K // 2
        self.padding_mode = padding_mode

        # 1) validate file tồn tại
        self._validate_files()

        # 2) load metadata (.pkl) nếu cần
        self.properties_cached: List[Optional[Dict]] = []
        if self.use_properties:
            print("Loading properties (.pkl)...")
            for p in tqdm(self.datalist, desc="Loading metadata"):
                props = self._load_pkl(p)
                self.properties_cached.append(props)
        else:
            self.properties_cached = [None] * len(self.datalist)

        # 3) chuẩn bị thông tin volume (chỉ đọc shape, không load full)
        #    + xây dựng list sample (case_idx, z_idx)
        print("Indexing volumes & building 2.5D slice list...")
        self.volume_shapes: List[Tuple[int, int, int, int]] = []  # list of (C,D,H,W)
        self.samples: List[Tuple[int, int]] = []                  # (case_idx, z)

        for case_idx, p in enumerate(tqdm(self.datalist, desc="Scanning volumes")):
            image_path = p.replace(".npz", ".npy")
            # dùng mmap để đọc shape nhanh
            img_mmap = np.load(image_path, mmap_mode="r")
            if img_mmap.ndim == 3:
                # (D,H,W) -> giả định C=1
                D, H, W = img_mmap.shape
                C = 1
            else:
                # (C,D,H,W)
                C, D, H, W = img_mmap.shape

            self.volume_shapes.append((C, D, H, W))

            # ===== THAY ĐỔI: Lấy TẤT CẢ slices, không bỏ biên =====
            for z in range(D):
                self.samples.append((case_idx, z))

        # 4) optional: cache full volume image+seg vào RAM
        self.volume_cache = {}
        if self.cache_volumes:
            print("Caching full volumes to RAM (image & seg)...")
            for case_idx, p in enumerate(tqdm(self.datalist, desc="Caching volumes")):
                img, seg = self._read_full_volume(p)
                self.volume_cache[case_idx] = (img, seg)

        print(f"Seg2p5DMedicalDataset initialized with {len(self.samples)} slice samples "
              f"from {len(self.datalist)} volumes.")
        print(f"Padding mode: {self.padding_mode}")

    # ---------- helper functions ----------

    def _validate_files(self):
        for data_path in self.datalist:
            image_path = data_path.replace(".npz", ".npy")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            if self.use_properties:
                pkl_path = data_path.replace(".npz", ".pkl")
                if not os.path.exists(pkl_path):
                    raise FileNotFoundError(f"Properties file not found: {pkl_path}")

            if not self.test:
                seg_path = data_path.replace(".npz", "_seg.npy")
                if not os.path.exists(seg_path):
                    raise FileNotFoundError(f"Segmentation file not found: {seg_path}")

    def _load_pkl(self, data_path: str) -> Dict:
        properties_path = data_path.replace(".npz", ".pkl")
        with open(properties_path, "rb") as f:
            props = pickle.load(f)
        return props

    def _read_full_volume(self, data_path: str):
        """Đọc full image & seg vào RAM (chỉ dùng khi cache_volumes=True)."""
        image_path = data_path.replace(".npz", ".npy")
        img = np.load(image_path)  # load đầy đủ

        seg = None
        if not self.test:
            seg_path = data_path.replace(".npz", "_seg.npy")
            seg = np.load(seg_path)

        # chuẩn hóa shape theo giả định
        if img.ndim == 3:   # (D,H,W) -> (1,D,H,W)
            img = img[None, ...]
        if (seg is not None) and (seg.ndim == 4):  # (1,D,H,W) -> (D,H,W)
            seg = seg[0]

        return img, seg

    def _read_volume_mmap(self, data_path: str):
        """Đọc volume bằng mmap (không cache, dùng mỗi __getitem__)."""
        image_path = data_path.replace(".npz", ".npy")
        img = np.load(image_path, mmap_mode="r")

        seg = None
        if not self.test:
            seg_path = data_path.replace(".npz", "_seg.npy")
            seg = np.load(seg_path, mmap_mode="r")

        if img.ndim == 3:
            img = img[None, ...]  # (1,D,H,W)
        if (seg is not None) and (seg.ndim == 4):
            seg = seg[0]          # (D,H,W)

        return img, seg

    def _get_slice(self, vol: np.ndarray, z: int):
        """
        vol: image (C,D,H,W) hoặc seg (D,H,W)
        trả về:
        - image slice: (C,H,W)
        - seg slice  : (H,W)
        """
        if vol.ndim == 4:
            # image: (C,D,H,W)
            return vol[:, z, :, :]
        else:
            # seg: (D,H,W)
            return vol[z, :, :]

    def _get_neighbor_index(self, z: int, offset: int, D: int) -> int:
        """
        Tính index của neighbor slice với padding strategy
        
        Args:
            z: center slice index
            offset: offset từ center (-half đến +half, trừ 0)
            D: tổng số slices trong volume
        
        Returns:
            z_neighbor: index của neighbor slice (đã xử lý boundary)
        """
        z_n = z + offset
        
        if self.padding_mode == "reflect":
            # Reflection padding: 0 1 2 3 → 3 2 1 | 0 1 2 3 | 2 1 0
            if z_n < 0:
                z_n = -z_n  # reflect từ boundary trái
            elif z_n >= D:
                z_n = 2 * D - z_n - 2  # reflect từ boundary phải
            z_n = np.clip(z_n, 0, D - 1)  # ensure in bounds
            
        elif self.padding_mode == "replicate":
            # Replication padding: 0 0 0 | 0 1 2 3 | 3 3 3
            z_n = np.clip(z_n, 0, D - 1)
            
        elif self.padding_mode == "zero":
            # Zero padding: sẽ trả về slice zero nếu out of bounds
            # (xử lý trong _get_slice_safe)
            pass
        
        else:
            raise ValueError(f"Unknown padding mode: {self.padding_mode}")
        
        return z_n

    def _get_slice_safe(self, vol: np.ndarray, z: int, D: int):
        """
        Get slice với xử lý zero padding nếu z out of bounds
        """
        if z < 0 or z >= D:
            # Return zero slice
            if vol.ndim == 4:
                C, _, H, W = vol.shape
                return np.zeros((C, H, W), dtype=vol.dtype)
            else:
                _, H, W = vol.shape
                return np.zeros((H, W), dtype=vol.dtype)
        else:
            return self._get_slice(vol, z)

    # ---------- Dataset API ----------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        case_idx, z = self.samples[idx]
        data_path = self.datalist[case_idx]

        # 1) lấy volume từ cache hoặc mmap
        if self.cache_volumes:
            img_vol, seg_vol = self.volume_cache[case_idx]
        else:
            img_vol, seg_vol = self._read_volume_mmap(data_path)

        # Get depth dimension
        if img_vol.ndim == 4:
            C, D, H, W = img_vol.shape
        else:
            D, H, W = img_vol.shape
            C = 1

        # 2) lấy center
        center_img = self._get_slice(img_vol, z)          # (C,H,W)
        center_seg = None if self.test else self._get_slice(seg_vol, z)  # (H,W)

        # 3) lấy neighbors với padding
        neighbor_slices = []
        for offset in range(-self.half, self.half + 1):
            if offset == 0:
                continue
            
            # ===== THAY ĐỔI: Xử lý boundary với padding =====
            if self.padding_mode == "zero":
                # Zero padding: trả về zero slice nếu out of bounds
                z_n = z + offset
                neighbor_slice = self._get_slice_safe(img_vol, z_n, D)
            else:
                # Reflect/Replicate padding
                z_n = self._get_neighbor_index(z, offset, D)
                neighbor_slice = self._get_slice(img_vol, z_n)
            
            neighbor_slices.append(neighbor_slice)  # (C,H,W)
        
        neighbors = np.stack(neighbor_slices, axis=0)  # (K, C, H, W)

        # 4) chuyển sang torch
        center_img = torch.from_numpy(center_img.copy()).float().contiguous()       # (C,H,W)
        neighbors = torch.from_numpy(neighbors.copy()).float().contiguous()         # (K,C,H,W)
        if center_seg is not None:
            center_seg = torch.from_numpy(center_seg.copy()).long().contiguous()    # (H,W)

        properties = self.properties_cached[case_idx] if self.use_properties else None

        # 5) preprocess/augment (nếu có)
        if self.preprocess_fn is not None:
            center_img, neighbors, center_seg = self.preprocess_fn(
                center_img, neighbors, center_seg
            )

        return {
            "center": center_img,
            "neighbors": neighbors,
            "mask": center_seg,     # None khi test=True
            "properties": properties,
            "case_idx": case_idx,
            "z_idx": z,
            "global_idx": idx,
        }