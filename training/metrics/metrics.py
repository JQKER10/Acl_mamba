import torch
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import List, Dict


def dice_coefficient(pred, target, smooth=1e-6, threshold=0.5):
    """
    Calculate Dice Coefficient
    Args:
        pred: [B, C, H, W] - predicted logits
        target: [B, C, H, W] - ground truth
        smooth: smoothing factor
        threshold: threshold for binary prediction
    Returns:
        Dice score
    """
    pred = torch.softmax(pred, dim=1)
    
    # Binarize predictions
    pred_binary = (pred > threshold).float()
    
    # Flatten
    pred_flat = pred_binary.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    # Calculate dice
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice


def dice_coefficient_per_class(pred, target, num_classes, smooth=1e-6, ignore_background=True):
    """
    Calculate Dice Coefficient per class
    Returns:
        List of dice scores for each class
    """
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)  # [B, H, W]
    target = torch.argmax(target, dim=1)  # [B, H, W]
    
    dice_scores = []
    start_idx = 1 if ignore_background else 0
    
    for c in range(start_idx, num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        pred_flat = pred_c.view(-1)
        target_flat = target_c.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_scores.append(dice.item())
    
    return dice_scores


def iou_score(pred, target, smooth=1e-6, threshold=0.5):
    """
    Calculate IoU (Intersection over Union) / Jaccard Index
    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
    Returns:
        IoU score
    """
    pred = torch.softmax(pred, dim=1)
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def pixel_accuracy(pred, target):
    """
    Calculate pixel-wise accuracy
    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
    Returns:
        Accuracy percentage
    """
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)  # [B, H, W]
    target = torch.argmax(target, dim=1)  # [B, H, W]
    
    correct = (pred == target).sum().float()
    total = target.numel()
    
    accuracy = correct / total
    
    return accuracy


def precision_recall_f1(pred, target, threshold=0.5):
    """
    Calculate Precision, Recall, and F1 Score
    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
    Returns:
        precision, recall, f1 score
    """
    pred = torch.softmax(pred, dim=1)
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    TP = (pred_flat * target_flat).sum()
    FP = (pred_flat * (1 - target_flat)).sum()
    FN = ((1 - pred_flat) * target_flat).sum()
    
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return precision, recall, f1


def sensitivity_specificity(pred, target, threshold=0.5):
    """
    Calculate Sensitivity (Recall/TPR) and Specificity (TNR)
    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
    Returns:
        sensitivity, specificity
    """
    pred = torch.softmax(pred, dim=1)
    pred_binary = (pred > threshold).float()
    
    pred_flat = pred_binary.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    
    TP = (pred_flat * target_flat).sum()
    TN = ((1 - pred_flat) * (1 - target_flat)).sum()
    FP = (pred_flat * (1 - target_flat)).sum()
    FN = ((1 - pred_flat) * target_flat).sum()
    
    sensitivity = TP / (TP + FN + 1e-6)  # True Positive Rate
    specificity = TN / (TN + FP + 1e-6)  # True Negative Rate
    
    return sensitivity, specificity


def hausdorff_distance(pred, target, threshold=0.5):
    """
    Calculate Hausdorff Distance (95th percentile)
    Note: This is computationally expensive, use sparingly
    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
    Returns:
        Hausdorff distance (HD95)
    """
    pred = torch.softmax(pred, dim=1)
    pred_binary = (pred > threshold).cpu().numpy()
    target_binary = target.cpu().numpy()
    
    batch_size = pred.shape[0]
    hd_list = []
    
    for b in range(batch_size):
        pred_b = pred_binary[b, 1, :, :]  # Foreground class
        target_b = target_binary[b, 1, :, :]
        
        # Skip if either mask is empty
        if pred_b.sum() == 0 or target_b.sum() == 0:
            hd_list.append(np.inf)
            continue
        
        # Distance transform
        pred_dt = distance_transform_edt(~pred_b.astype(bool))
        target_dt = distance_transform_edt(~target_b.astype(bool))
        
        # Get surface points
        pred_surface = pred_b.astype(bool)
        target_surface = target_b.astype(bool)
        
        # Calculate distances
        distances_pred_to_target = target_dt[pred_surface]
        distances_target_to_pred = pred_dt[target_surface]
        
        # Combine distances
        all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
        
        # 95th percentile
        hd95 = np.percentile(all_distances, 95)
        hd_list.append(hd95)
    
    return np.mean(hd_list)


def average_surface_distance(pred, target, threshold=0.5):
    """
    Calculate Average Surface Distance (ASD)
    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
    Returns:
        Average surface distance
    """
    pred = torch.softmax(pred, dim=1)
    pred_binary = (pred > threshold).cpu().numpy()
    target_binary = target.cpu().numpy()
    
    batch_size = pred.shape[0]
    asd_list = []
    
    for b in range(batch_size):
        pred_b = pred_binary[b, 1, :, :]
        target_b = target_binary[b, 1, :, :]
        
        if pred_b.sum() == 0 or target_b.sum() == 0:
            asd_list.append(np.inf)
            continue
        
        pred_dt = distance_transform_edt(~pred_b.astype(bool))
        target_dt = distance_transform_edt(~target_b.astype(bool))
        
        pred_surface = pred_b.astype(bool)
        target_surface = target_b.astype(bool)
        
        distances_pred = target_dt[pred_surface]
        distances_target = pred_dt[target_surface]
        
        asd = (distances_pred.sum() + distances_target.sum()) / (len(distances_pred) + len(distances_target))
        asd_list.append(asd)
    
    return np.mean(asd_list)


def compute_all_metrics(pred, target, num_classes=2):
    """
    Compute all metrics at once
    Args:
        pred: [B, C, H, W]
        target: [B, C, H, W]
        num_classes: number of classes
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['dice'] = dice_coefficient(pred, target).item()
    metrics['iou'] = iou_score(pred, target).item()
    metrics['pixel_acc'] = pixel_accuracy(pred, target).item()
    
    # Precision, Recall, F1
    precision, recall, f1 = precision_recall_f1(pred, target)
    metrics['precision'] = precision.item()
    metrics['recall'] = recall.item()
    metrics['f1'] = f1.item()
    
    # Sensitivity, Specificity
    sensitivity, specificity = sensitivity_specificity(pred, target)
    metrics['sensitivity'] = sensitivity.item()
    metrics['specificity'] = specificity.item()
    
    # Per-class dice
    dice_per_class = dice_coefficient_per_class(pred, target, num_classes)
    for i, dice in enumerate(dice_per_class):
        metrics[f'dice_class_{i+1}'] = dice
    
    return metrics




# ==================== SEGMENTATION METRICS ====================

class SegmentationMetrics:
    """
    Basic segmentation metrics (2D):
      - Dice per class + mean
      - IoU per class + mean
      - Precision / Recall / Specificity per class
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -100,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.long,
        )
        self.dice_scores: List[List[float]] = []
        self.iou_scores: List[List[float]] = []

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Args:
            pred:   [B,C,H,W] logits or [B,H,W] predictions
            target: [B,H,W] labels
        """
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)  # [B,H,W]

        pred = pred.view(-1).cpu()
        target = target.view(-1).cpu()

        # remove ignore_index
        if self.ignore_index >= 0:
            mask = target != self.ignore_index
            pred = pred[mask]
            target = target[mask]

        # update confusion matrix vectorized
        k = (target * self.num_classes + pred).long()
        bins = torch.bincount(
            k, minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        self.confusion_matrix += bins

        # per-class dice/iou for this batch (flattened)
        dice_per_class = self._compute_dice_per_class(pred, target)
        iou_per_class = self._compute_iou_per_class(pred, target)

        self.dice_scores.append(dice_per_class)
        self.iou_scores.append(iou_per_class)

    def _compute_dice_per_class(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> List[float]:
        dice_scores = []
        for c in range(self.num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()

            inter = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            if union > 0:
                dice = (2.0 * inter / union).item()
                dice_scores.append(dice)
            else:
                dice_scores.append(np.nan)
        return dice_scores

    def _compute_iou_per_class(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> List[float]:
        iou_scores = []
        for c in range(self.num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()

            inter = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum() - inter

            if union > 0:
                iou = (inter / union).item()
                iou_scores.append(iou)
            else:
                iou_scores.append(np.nan)
        return iou_scores

    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Dice
        dice_array = np.array(self.dice_scores)  # [N, C]
        for c in range(self.num_classes):
            dice_c = dice_array[:, c]
            dice_c = dice_c[~np.isnan(dice_c)]
            if len(dice_c) > 0:
                metrics[f"dice_class_{c}"] = float(np.mean(dice_c))
        valid_dice = dice_array[~np.isnan(dice_array)]
        metrics["mean_dice"] = float(np.mean(valid_dice)) if len(valid_dice) > 0 else 0.0

        # IoU
        iou_array = np.array(self.iou_scores)
        for c in range(self.num_classes):
            iou_c = iou_array[:, c]
            iou_c = iou_c[~np.isnan(iou_c)]
            if len(iou_c) > 0:
                metrics[f"iou_class_{c}"] = float(np.mean(iou_c))
        valid_iou = iou_array[~np.isnan(iou_array)]
        metrics["mean_iou"] = float(np.mean(valid_iou)) if len(valid_iou) > 0 else 0.0

        # Precision / Recall / Specificity from confusion matrix
        cm = self.confusion_matrix.float()
        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

            metrics[f"precision_class_{c}"] = float(precision)
            metrics[f"recall_class_{c}"] = float(recall)
            metrics[f"specificity_class_{c}"] = float(specificity)

        return metrics

    def summary(self) -> str:
        m = self.compute()
        lines = ["Segmentation Metrics", "=" * 50]

        lines.append("Dice:")
        for c in range(self.num_classes):
            key = f"dice_class_{c}"
            if key in m:
                lines.append(f"  Class {c}: {m[key]:.4f}")
        lines.append(f"  Mean: {m['mean_dice']:.4f}")

        lines.append("\nIoU:")
        for c in range(self.num_classes):
            key = f"iou_class_{c}"
            if key in m:
                lines.append(f"  Class {c}: {m[key]:.4f}")
        lines.append(f"  Mean: {m['mean_iou']:.4f}")

        lines.append("=" * 50)
        return "\n".join(lines)

