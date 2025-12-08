import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch

class Medical2p5DTransform:
    def __init__(self, image_size: int = 256, is_train: bool = True):
        self.is_train = is_train
        
        # Các transform hình học (Geometric) - Áp dụng cho cả Ảnh và Mask
        # Quan trọng: Elastic/Grid Distortion rất tốt cho mô mềm (sụn, dây chằng)
        train_geo_transforms = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
            A.CoarseDropout(max_holes=8, max_height=int(image_size/10), max_width=int(image_size/10), p=0.2),
        ]
        
        # Các transform màu sắc/nhiễu (Pixel-level) - CHỈ áp dụng cho Ảnh
        # Medical images thường nhạy cảm với brightness/contrast
        train_pixel_transforms = [
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ]

        # Pipeline
        if self.is_train:
            self.transform = A.Compose(
                train_geo_transforms + train_pixel_transforms + [
                    A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0), # Cần điều chỉnh theo dataset của bạn
                    ToTensorV2()
                ]
            )
        else:
            # Validation chỉ Normalize
            self.transform = A.Compose([
                A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=1.0),
                ToTensorV2()
            ])

    def __call__(self, img_tensor: torch.Tensor, mask_tensor: torch.Tensor = None):
        """
        Args:
            img_tensor: Tensor shape (C, H, W) - đây là stack 2.5D
            mask_tensor: Tensor shape (H, W) hoặc None
        Returns:
            img_aug, mask_aug (Tensors)
        """
        # 1. Convert Tensor (C, H, W) -> Numpy (H, W, C) để Albumentations hiểu
        # Albumentations coi các slice lân cận như là các kênh màu (RGB channels)
        img_np = img_tensor.permute(1, 2, 0).numpy() 
        
        # Chuyển về float32 nếu chưa phải (để tránh lỗi augment)
        img_np = img_np.astype(np.float32)

        # 2. Chuẩn bị mask
        mask_np = None
        if mask_tensor is not None:
            mask_np = mask_tensor.numpy().astype(np.float32)

        # 3. Apply Transform
        if mask_np is not None:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_aug = augmented['image'] # Đã là Tensor (C, H, W) nhờ ToTensorV2
            mask_aug = augmented['mask'] # Đã là Tensor
            
            # Mask output từ ToTensorV2 thường là float, convert về long cho loss function
            mask_aug = mask_aug.long()
        else:
            augmented = self.transform(image=img_np)
            img_aug = augmented['image']
            mask_aug = None

        return img_aug, mask_aug