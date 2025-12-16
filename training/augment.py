import random
from typing import Tuple, Optional, List, Any

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --- Custom Transforms ---

class RicianNoise(A.ImageOnlyTransform):
    """Giả lập nhiễu Rician đặc trưng của Medical Imaging."""
    def __init__(self, noise_limit: Tuple[float, float] = (0.01, 0.05), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.noise_limit = noise_limit

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        variance = np.random.uniform(*self.noise_limit)
        sigma = np.sqrt(variance)
        
        n1 = np.random.normal(0, sigma, img.shape)
        n2 = np.random.normal(0, sigma, img.shape)
        
        img_rician = np.sqrt((img + n1) ** 2 + n2 ** 2)
        return np.clip(img_rician, 0, 1).astype(img.dtype)

    def get_transform_init_args_names(self):
        return ("noise_limit",)


# --- Main Pipeline ---

class Medical2p5DTransform:
    def __init__(
        self, 
        image_size: int = 256, 
        is_train: bool = True,
        mean: Tuple[float] = (0.5,),
        std: Tuple[float] = (0.5,),
        use_normalization: bool = True
    ):
        self.image_size = image_size
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.use_normalization = use_normalization

        # Augmentation components
        self.spatial_transform = self._build_spatial() if is_train else None
        self.pixel_transform = self._build_pixel() if is_train else None
        self.final_transform = self._build_final()

    def _build_spatial(self) -> A.Compose:

        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.05, 0.05), # Dịch chuyển cả 2 chiều
                rotate=(-15, 15),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST, 
                mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            
            A.OneOf([
                A.GridDistortion(
                    num_steps=5, distort_limit=0.3, 
                    interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST, 
                    p=1.0
                ),
                A.ElasticTransform(
                    alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03,
                    interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0
                ),
            ], p=0.3),

            A.CoarseDropout(
                max_holes=8,
                max_height=int(self.image_size / 10),
                max_width=int(self.image_size / 10),
                p=0.2
            ),
        ])

    def _build_pixel(self) -> A.Compose:
        """Intensity transforms: Brightness, Noise, Gamma..."""
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=0.2),
            
            A.OneOf([
                A.GaussNoise(var_limit=(0.001, 0.01), p=1.0),
                RicianNoise(noise_limit=(0.001, 0.01), p=1.0),
            ], p=0.3),
            
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ])

    def _build_final(self) -> A.Compose:
        transforms = []
        if self.use_normalization:
            transforms.append(
                A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0)
            )
        transforms.append(ToTensorV2())
        return A.Compose(transforms)

    def __call__(
        self, 
        center_img: torch.Tensor, 
        neighbors: torch.Tensor,
        center_seg: Optional[torch.Tensor] = None
    ):
        # Detach về CPU để xử lý
        if isinstance(center_img, torch.Tensor): center_img = center_img.detach().cpu()
        if isinstance(neighbors, torch.Tensor): neighbors = neighbors.detach().cpu()
            
        C, H, W = center_img.shape
        K = neighbors.shape[0]

        # 1. Stack Channels -> (H, W, Total_Channels)
        neighbors_flat = neighbors.view(-1, H, W) 
        full_vol = torch.cat([center_img, neighbors_flat], dim=0)
        img_np = full_vol.permute(1, 2, 0).numpy().astype(np.float32)

       
        img_np = np.nan_to_num(img_np, nan=0.0, posinf=None, neginf=None)

     
        _min = img_np.min()
        _max = img_np.max()
        if _max - _min > 1e-5:
            img_np = (img_np - _min) / (_max - _min)
        else:
            img_np = np.zeros_like(img_np)
       
        mask_np = None
        if center_seg is not None:
            if isinstance(center_seg, torch.Tensor): center_seg = center_seg.detach().cpu()
            mask_np = center_seg.numpy().astype(np.int32) 

        # 2. Augmentation Flow
        if self.is_train:
            if mask_np is not None:
                augmented = self.spatial_transform(image=img_np, mask=mask_np)
                img_np, mask_np = augmented['image'], augmented['mask']
            else:
                img_np = self.spatial_transform(image=img_np)['image']

            img_np = self.pixel_transform(image=img_np)['image']

        # 3. Finalize
        if mask_np is not None:
            final = self.final_transform(image=img_np, mask=mask_np)
            img_tensor = final['image']
            mask_tensor = final['mask'].long()
        else:
            final = self.final_transform(image=img_np)
            img_tensor = final['image']
            mask_tensor = None
            
        # Kiểm tra an toàn lần cuối
        if torch.isnan(img_tensor).any():
            img_tensor = torch.nan_to_num(img_tensor, nan=0.0)

        # 4. Unstack channels
        center_img_aug = img_tensor[:C, ...]
        neighbors_flat_aug = img_tensor[C:, ...]
        neighbors_aug = neighbors_flat_aug.view(K, C, H, W) # Giữ nguyên H, W

        return center_img_aug, neighbors_aug, mask_tensor

    def __repr__(self):
        mode = "Train" if self.is_train else "Val"
        return f"Medical2p5DTransform(mode={mode}, size={self.image_size}, norm={self.use_normalization})"

