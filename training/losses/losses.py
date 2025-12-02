import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Optional, List, Dict, Union, Callable


# ==================== DICE LOSS ====================

class DiceLoss(nn.Module):
    """
    Multi-class Dice loss with optional ignore_index and optional exclusion of background.
    """
    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = -100,
        include_background: bool = True,
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.include_background = include_background

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, C, H, W] logits
            target: [B, H, W] class labels
        """
        # Softmax over channels
        pred = torch.softmax(pred, dim=1)  # [B,C,H,W]
        num_classes = pred.shape[1]

        # Mask ignore_index
        if self.ignore_index >= 0:
            valid_mask = (target != self.ignore_index)  # [B,H,W]
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool)

        # Clamp target for one_hot to avoid out-of-range index
        target_clamped = target.clone()
        target_clamped[~valid_mask] = 0  # tạm đặt background, sẽ mask lại sau

        # One-hot target: [B,H,W,C] -> [B,C,H,W]
        target_one_hot = F.one_hot(target_clamped.long(), num_classes=num_classes)  # [B,H,W,C]
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()                # [B,C,H,W]

        # Áp mask (nhân 0 những pixel ignore)
        valid_mask = valid_mask.unsqueeze(1)  # [B,1,H,W]
        pred = pred * valid_mask
        target_one_hot = target_one_hot * valid_mask

        dice_scores = []

        class_range = range(num_classes)
        if not self.include_background and num_classes > 1:
            class_range = range(1, num_classes)

        for c in class_range:
            pred_c = pred[:, c]           # [B,H,W]
            target_c = target_one_hot[:, c]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)

        if len(dice_scores) == 0:
            # tất cả class bị bỏ → trả 0 để tránh NaN
            return pred.new_tensor(0.0)

        dice_loss = 1.0 - torch.stack(dice_scores).mean()
        return dice_loss


# ==================== FOCAL LOSS ====================

class FocalLoss(nn.Module):
    """
    Multi-class focal loss (on top of cross-entropy).
    """
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, C, H, W] logits
            target: [B, H, W] labels
        """
        ce_loss = F.cross_entropy(
            pred,
            target,
            reduction="none",
            ignore_index=self.ignore_index,
        )  # [B,H,W]

        # p_t
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        return focal_loss.mean()


# ==================== DEEP SUPERVISION WRAPPER ====================

class DeepSupervisionLoss(nn.Module):
    """
    Wrapper for deep supervision:
      - base_loss: Callable(pred, target) -> Dict[str, Tensor] (must contain 'total')
      - outputs: list of predictions at different scales
      - targets được downsample để khớp với từng output
    """

    def __init__(
        self,
        base_loss: Callable[[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]],
        num_deep_supervision: int = 3,
        weights: Optional[List[float]] = None,
        downsample_mode: str = "nearest",
    ):
        """
        Args:
            base_loss:     Callable that returns a dict with key 'total'
            num_deep_supervision: number of deep supervision outputs (ds1..dsn),
                                 main output tính là 0 → tổng là n+1 output
            weights:       weights for [main, ds1, ds2, ...], sẽ được normalize
            downsample_mode: 'nearest' cho segmentation
        """
        super().__init__()
        self.base_loss = base_loss
        self.num_deep_supervision = num_deep_supervision
        self.downsample_mode = downsample_mode

        if weights is None:
            # 1.0, 0.5, 0.25, ... (n+1 phần tử)
            self.weights = [1.0 / (2 ** i) for i in range(num_deep_supervision + 1)]
        else:
            assert len(weights) == num_deep_supervision + 1, \
                f"weights length must be {num_deep_supervision + 1}"
            self.weights = weights

        # Normalize
        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]

        print("DeepSupervisionLoss initialized")
        print(f"  num outputs (main + ds): {num_deep_supervision + 1}")
        print(f"  normalized weights: {self.weights}")

    def forward(
        self,
        outputs: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs:
                - Tensor: [B,C,H,W] (no deep supervision)
                - List:   [main, ds1, ds2, ...]
            target: [B,H,W]
        """
        # Không dùng deep supervision
        if isinstance(outputs, torch.Tensor):
            loss_dict = self.base_loss(outputs, target)
            if "total" not in loss_dict:
                # fallback nếu base_loss chỉ trả 1 scalar
                loss_dict = {"total": loss_dict["loss"]}
            return loss_dict

        # Có deep supervision
        assert isinstance(outputs, (list, tuple)), \
            "For deep supervision, outputs must be list/tuple"

        total_loss = 0.0
        loss_dict: Dict[str, torch.Tensor] = {}

        for i, output in enumerate(outputs):
            # Downsample target nếu cần
            if output.shape[-2:] != target.shape[-2:]:
                target_ds = F.interpolate(
                    target.unsqueeze(1).float(),   # [B,1,H,W]
                    size=output.shape[-2:],        # (H_i, W_i)
                    mode=self.downsample_mode,
                ).squeeze(1).long()               # [B,H_i,W_i]
            else:
                target_ds = target

            out_loss_dict = self.base_loss(output, target_ds)

            if "total" in out_loss_dict:
                out_loss = out_loss_dict["total"]
            elif "loss" in out_loss_dict:
                out_loss = out_loss_dict["loss"]
            else:
                raise ValueError("base_loss must return dict with key 'total' or 'loss'.")

            weighted = self.weights[i] * out_loss
            total_loss = total_loss + weighted

            if i == 0:
                loss_dict["main_loss"] = out_loss
                # copy các thành phần của main loss
                for k, v in out_loss_dict.items():
                    if k != "total":
                        loss_dict[k] = v
            else:
                loss_dict[f"ds_loss_{i}"] = out_loss

        loss_dict["total"] = total_loss
        return loss_dict


# ==================== HYBRID LOSS + DEEP SUPERVISION ====================

class HybridLossWithDeepSupervision(nn.Module):
    """
    Hybrid loss = Dice + CrossEntropy + (optional) Focal
    + deep supervision wrapper.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        focal_weight: float = 0.0,  # nên bắt đầu = 0.0 hoặc nhỏ
        num_deep_supervision: int = 3,
        ds_weights: Optional[List[float]] = None,
        ignore_index: int = -100,
        include_background: bool = True,
    ):
        super().__init__()

        # Base losses
        self.dice_loss = DiceLoss(
            ignore_index=ignore_index,
            include_background=include_background,
        )
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss(ignore_index=ignore_index)

        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

        # base hybrid loss function
        base_loss = self._hybrid_loss_fn

        # deep supervision wrapper
        self.deep_supervision = DeepSupervisionLoss(
            base_loss=base_loss,
            num_deep_supervision=num_deep_supervision,
            weights=ds_weights,
        )

        print("HybridLossWithDeepSupervision initialized")
        print(f"  dice_weight  = {dice_weight}")
        print(f"  ce_weight    = {ce_weight}")
        print(f"  focal_weight = {focal_weight}")

    def _hybrid_loss_fn(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        pred:   [B,C,H,W]
        target: [B,H,W]
        """
        dice = self.dice_loss(pred, target)
        ce = self.ce_loss(pred, target)

        if self.focal_weight > 0.0:
            focal = self.focal_loss(pred, target)
        else:
            focal = pred.new_tensor(0.0)

        total = (
            self.dice_weight * dice
            + self.ce_weight * ce
            + self.focal_weight * focal
        )

        return {
            "total": total,
            "dice": dice,
            "ce": ce,
            "focal": focal,
        }

    def forward(
        self,
        outputs: Union[torch.Tensor, List[torch.Tensor]],
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.deep_supervision(outputs, target)


