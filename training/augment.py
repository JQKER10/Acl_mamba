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
    Công thức: I_out = sqrt((I + N1)^2 + N2^2)
    
    Attributes:
        noise_limit (Tuple[float, float]): Phạm vi variance của nhiễu.
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
        # img shape: (H, W, C), values: [0, 1] usually
        variance = np.random.uniform(self.noise_limit[0], self.noise_limit[1])
        sigma = np.sqrt(variance)
        
        # Tạo 2 mẫu nhiễu Gaussian độc lập
        n1 = np.random.normal(0, sigma, img.shape)
        n2 = np.random.normal(0, sigma, img.shape)
        
        # Áp dụng công thức Rician
        img_rician = np.sqrt((img + n1) ** 2 + n2 ** 2)
        
        # Đảm bảo dtype và range không thay đổi
        return np.clip(img_rician, 0, 1).astype(img.dtype)

    def get_transform_init_args_names(self):
        return ("noise_limit",)


class SimulateRegistrationError(ImageOnlyTransform):
    """
    Mô phỏng lỗi lệch ghép kênh (Channel Misalignment/Translation).
    Dịch chuyển ngẫu nhiên các slice (channels) độc lập với nhau.
    
    Attributes:
        shift_limit (int): Số pixel tối đa dịch chuyển theo trục x, y.
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
        
        # Dịch chuyển từng kênh riêng biệt
        for i in range(c):
            dx = random.randint(-self.shift_limit, self.shift_limit)
            dy = random.randint(-self.shift_limit, self.shift_limit)
            
            # Sử dụng Affine matrix để dịch chuyển
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            
            # BORDER_REFLECT_101 giúp biên ảnh mượt hơn khi dịch chuyển
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
    Pipeline Augmentation chuyên dụng cho dữ liệu Y tế 2.5D (Stack slices).
    Input: Tensor (C, H, W) -> Output: Tensor (C, H, W)
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
        """Nhóm biến đổi hình học (Geometric): Xoay, lật, méo."""
        return [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5), # Tùy chọn, tốt cho MRI đầu gối/cột sống
            
            # Affine thay thế cho ShiftScaleRotate (mạnh mẽ hơn)
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(0, 0.05),
                rotate=(-15, 15),
                interpolation=cv2.INTER_LINEAR,
                mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            
            # Biến dạng đàn hồi (Elastic/Grid) - Rất quan trọng cho mô mềm
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.ElasticTransform(
                    alpha=120, 
                    sigma=120 * 0.05, 
                    alpha_affine=120 * 0.03, 
                    p=1.0
                ),
            ], p=0.3),

            # Giả lập lỗi ghép kênh (Registration Error)
            SimulateRegistrationError(shift_limit=3, p=0.2),
            
            # CoarseDropout - Giả lập mất thông tin cục bộ
            A.CoarseDropout(
                max_holes=8,
                max_height=int(self.image_size / 10),
                max_width=int(self.image_size / 10),
                p=0.2
            ),
        ]

    def _get_pixel_transforms(self) -> List[Any]:
        """Nhóm biến đổi pixel (Intensity/Noise): Màu sắc, độ sáng, nhiễu."""
        return [
            # 1. Intensity Corrections
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.3
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=0.2),
            
            # 2. Noise Injection
            A.OneOf([
                A.GaussNoise(var_limit=(0.001, 0.01), p=1.0),
                RicianNoise(noise_limit=(0.001, 0.01), p=1.0),
            ], p=0.3),
            
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ]

    def _get_train_pipeline(self) -> A.Compose:
        """Tổng hợp pipeline cho training."""
        transforms_list = (
            self._get_spatial_transforms() + 
            self._get_pixel_transforms() + 
            [
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0),
                ToTensorV2()
            ]
        )
        return A.Compose(transforms_list)

    def _get_val_pipeline(self) -> A.Compose:
        """Tổng hợp pipeline cho validation (chỉ Normalize)."""
        return A.Compose([
            A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0),
            ToTensorV2()
        ])

    def __call__(
        self, 
        img_tensor: torch.Tensor, 
        mask_tensor: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Thực thi transform.
        
        Args:
            img_tensor: Input Image Tensor (C, H, W).
            mask_tensor: Input Mask Tensor (H, W) hoặc None.
            
        Returns:
            Tuple[img_aug, mask_aug]
        """
        # 1. Tensor (C, H, W) -> Numpy (H, W, C)
        # Permute để đưa Channels về cuối phục vụ Albumentations
        img_np = img_tensor.permute(1, 2, 0).numpy().astype(np.float32)

        # 2. Mask Handling
        mask_np = None
        if mask_tensor is not None:
            mask_np = mask_tensor.numpy().astype(np.float32)

        # 3. Apply Transform Pipeline
        if mask_np is not None:
            # Albumentations tự động xử lý mask cho Spatial Transforms
            # và bỏ qua mask cho Pixel Transforms -> Rất an toàn
            augmented = self.transform(image=img_np, mask=mask_np)
            img_aug = augmented['image']
            mask_aug = augmented['mask'].long() # Mask segmentation cần kiểu Long/Int64
        else:
            augmented = self.transform(image=img_np)
            img_aug = augmented['image']
            mask_aug = None

        return img_aug, mask_aug

    def __repr__(self):
        """String representation cho debugging."""
        mode = "Train" if self.is_train else "Val"
        return f"Medical2p5DTransform(mode={mode}, size={self.image_size})"
