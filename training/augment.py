import random
from typing import Tuple, Optional, Union, Dict, Any, List

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

# =============================================================================
# CUSTOM TRANSFORMS
# =============================================================================

class RicianNoise(ImageOnlyTransform):
    """
    Mô phỏng nhiễu Rician đặc trưng của MRI.
    """
    def __init__(
        self, 
        noise_limit: Tuple[float, float] = (0.01, 0.05), 
        always_apply: bool = False, 
        p: float = 0.5
    ):
        super(RicianNoise, self).__init__(always_apply, p)
        self.noise_limit = noise_limit

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        # img shape: (H, W, Total_Channels)
        variance = np.random.uniform(self.noise_limit[0], self.noise_limit[1])
        sigma = np.sqrt(variance)
        
        n1 = np.random.normal(0, sigma, img.shape)
        n2 = np.random.normal(0, sigma, img.shape)
        
        img_rician = np.sqrt((img + n1) ** 2 + n2 ** 2)
        
        return np.clip(img_rician, 0, 1).astype(img.dtype)

    def get_transform_init_args_names(self):
        return ("noise_limit",)


class SimulateRegistrationError(ImageOnlyTransform):
    """
    Mô phỏng lỗi lệch ghép kênh.
    Lưu ý: Với 2.5D Stack, transform này sẽ làm lệch các neighbor so với center
    một cách ngẫu nhiên -> Tăng tính mạnh mẽ (Robustness).
    """
    def __init__(
        self, 
        shift_limit: int = 2, 
        always_apply: bool = False, 
        p: float = 0.5
    ):
        super(SimulateRegistrationError, self).__init__(always_apply, p)
        self.shift_limit = shift_limit

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w, c = img.shape
        img_shifted = np.empty_like(img)
        
        for i in range(c):
            dx = random.randint(-self.shift_limit, self.shift_limit)
            dy = random.randint(-self.shift_limit, self.shift_limit)
            
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            img_shifted[:, :, i] = cv2.warpAffine(
                img[:, :, i], M, (w, h), 
                borderMode=cv2.BORDER_REFLECT_101
            )
            
        return img_shifted

    def get_transform_init_args_names(self):
        return ("shift_limit",)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class Medical2p5DTransform:
    """
    Pipeline Augmentation chuyên dụng cho dữ liệu Y tế 2.5D.
    Hỗ trợ input từ Dataset: Center (C,H,W) + Neighbors (K,C,H,W).
    """
    def __init__(
        self, 
        image_size: int = 256, 
        is_train: bool = True,
        mean: Tuple[float] = (0.5,),
        std: Tuple[float] = (0.5,)
    ):
        self.image_size = image_size
        self.is_train = is_train
        self.mean = mean
        self.std = std
        
        if self.is_train:
            self.transform = self._get_train_pipeline()
        else:
            self.transform = self._get_val_pipeline()

    def _get_spatial_transforms(self) -> List[Any]:
        return [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(0, 0.05),
                rotate=(-15, 15),
                interpolation=cv2.INTER_LINEAR,
                mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.ElasticTransform(
                    alpha=120, 
                    sigma=120 * 0.05, 
                    alpha_affine=120 * 0.03, 
                    p=1.0
                ),
            ], p=0.3),

            # Simulation error chỉ nên tác động nhẹ, nếu dataset neighbors
            # đã được align chuẩn thì cái này giả lập chuyển động bệnh nhân tốt
            SimulateRegistrationError(shift_limit=3, p=0.2),
            
            A.CoarseDropout(
                max_holes=8,
                max_height=int(self.image_size / 10),
                max_width=int(self.image_size / 10),
                p=0.2
            ),
        ]

    def _get_pixel_transforms(self) -> List[Any]:
        return [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=0.2),
            
            A.OneOf([
                A.GaussNoise(var_limit=(0.001, 0.01), p=1.0),
                RicianNoise(noise_limit=(0.001, 0.01), p=1.0),
            ], p=0.3),
            
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ]

    def _get_train_pipeline(self) -> A.Compose:
        transforms_list = (
            self._get_spatial_transforms() + 
            self._get_pixel_transforms() + 
            [
                # Mean/Std sẽ tự broadcast nếu số channel > len(mean)
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0),
                ToTensorV2()
            ]
        )
        return A.Compose(transforms_list)

    def _get_val_pipeline(self) -> A.Compose:
        return A.Compose([
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0),
            ToTensorV2()
        ])

    def __call__(
        self, 
        center_img: torch.Tensor, 
        neighbors: torch.Tensor,
        center_seg: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Thực thi transform đồng bộ cho Center và Neighbors.
        
        Args:
            center_img: (C, H, W)
            neighbors:  (K, C, H, W)
            center_seg: (H, W) hoặc None
            
        Returns:
            Tuple[center_aug, neighbors_aug, mask_aug]
        """
        # 0. Lưu lại shape gốc để restore sau này
        C, H, W = center_img.shape
        K = neighbors.shape[0] # Số lượng neighbors

        # 1. Stack Center và Neighbors thành 1 khối (Channel Stacking)
        # neighbors (K, C, H, W) -> flatten -> (K*C, H, W)
        neighbors_flat = neighbors.view(-1, H, W) 
        
        # full_vol: (C + K*C, H, W)
        full_vol = torch.cat([center_img, neighbors_flat], dim=0)

        # 2. Convert sang Numpy (H, W, Total_Channels) cho Albumentations
        img_np = full_vol.permute(1, 2, 0).numpy().astype(np.float32)

        # 3. Mask Handling
        mask_np = None
        if center_seg is not None:
            mask_np = center_seg.numpy().astype(np.float32)

        # 4. Apply Transform Pipeline
        # Albumentations sẽ xử lý toàn bộ các kênh trong img_np với cùng một phép biến đổi hình học
        if mask_np is not None:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_aug = augmented['image']          # Tensor (Total_Channels, H, W)
            mask_aug = augmented['mask'].long()   # Tensor (H, W)
        else:
            augmented = self.transform(image=img_np)
            img_aug = augmented['image']
            mask_aug = None

        # 5. Unstack (Tách lại Center và Neighbors)
        # img_aug: (Total_Channels, H, W)
        
        # Lấy lại phần center (C channels đầu tiên)
        center_img_aug = img_aug[:C, :, :]
        
        # Lấy lại phần neighbors (phần còn lại)
        neighbors_flat_aug = img_aug[C:, :, :]
        
        # Reshape lại neighbors về (K, C, H, W)
        neighbors_aug = neighbors_flat_aug.view(K, C, H, W)

        return center_img_aug, neighbors_aug, mask_aug

    def __repr__(self):
        mode = "Train" if self.is_train else "Val"
        return f"Medical2p5DTransform(mode={mode}, size={self.image_size})"
