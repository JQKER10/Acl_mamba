# trainer_2p5d.py

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None


# ==================== LOGGING ====================

def setup_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # tránh add handlers nhiều lần
    if not logger.handlers:
        fmt = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger


# ==================== METRICS TRACKER ====================

class MetricTracker:
    """Theo dõi trung bình các metric theo step."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for k, v in metrics.items():
            if np.isnan(float(v)):
                continue
            self.sums[k] = self.sums.get(k, 0.0) + float(v)
            self.counts[k] = self.counts.get(k, 0) + 1

    def average(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, s in self.sums.items():
            c = self.counts.get(k, 0)
            out[k] = s / max(c, 1)
        return out

    def get_latest(self) -> Dict[str, float]:
        # ở đây dùng trung bình tạm thời luôn cho đơn giản
        return self.average()


# ==================== EARLY STOPPING ====================

class EarlyStopping:
    """Early stopping theo 1 metric."""

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = "max",
        verbose: bool = True,
    ) -> None:
        assert mode in ("max", "min")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_score: Optional[float] = None
        self.best_epoch: int = 0
        self.counter: int = 0
        self.early_stop: bool = False

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        improved = self._is_improvement(score)
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"✓ Validation improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"⚠ No improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(
                        f"⚠ Early stopping! Best {self.best_score:.4f} "
                        f"at epoch {self.best_epoch}"
                    )
        return self.early_stop

    def _is_improvement(self, score: float) -> bool:
        if self.mode == "max":
            return score > (self.best_score + self.min_delta)
        else:
            return score < (self.best_score - self.min_delta)


# ==================== CHECKPOINT MANAGER ====================

class CheckpointManager:
    """Quản lý save / xoá bớt checkpoint."""

    def __init__(
        self,
        save_dir: Union[str, Path],
        max_checkpoints: int = 3,
        monitor_metric: str = "val_dice",
        mode: str = "max",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.max_checkpoints = max_checkpoints
        self.monitor_metric = monitor_metric
        self.mode = mode

        self.best_score: Optional[float] = None
        self.checkpoint_files: List[Path] = []

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool,
    ) -> str:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "best_score": self.best_score,
        }

        path = self.save_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(ckpt, path)
        self.checkpoint_files.append(path)

        # xóa bớt cũ
        if len(self.checkpoint_files) > self.max_checkpoints:
            old = self.checkpoint_files.pop(0)
            if old.exists():
                old.unlink()

        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(ckpt, best_path)
            self.best_score = metrics.get(self.monitor_metric, 0.0)

        # luôn lưu last
        last_path = self.save_dir / "last_model.pth"
        torch.save(ckpt, last_path)

        return str(path)

    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        self.best_score = ckpt.get("best_score", None)
        return ckpt


# ==================== TRAINER ====================

class Trainer2p5D:
    """
    Trainer cho 2.5D segmentation:
      - model(center, neighbors) -> Tensor hoặc List[Tensor] (deep supervision).
      - criterion(outputs, mask) -> dict có 'total', 'dice', 'ce', 'focal', ...
      - batch: {'center', 'neighbors', 'mask', ...}
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str = "cuda",
        output_dir: Union[str, Path] = "./checkpoints",
        scheduler: Optional[_LRScheduler] = None,
        use_amp: bool = True,
        gradient_clip_val: float = 1.0,
        log_every: int = 10,
        val_every: int = 1,
        early_stopping_patience: int = 20,
        max_checkpoints: int = 3,
        monitor_metric: str = "val_dice",
        mode: str = "max",
        use_ddp: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
        wandb_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.use_amp = use_amp and device.startswith("cuda")
        self.gradient_clip_val = gradient_clip_val
        self.log_every = log_every
        self.val_every = val_every

        self.use_ddp = use_ddp
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main_process = (not use_ddp) or (local_rank == 0)

        # WandB
        self.use_wandb = self.is_main_process and wandb is not None and wandb_cfg is not None
        if self.use_wandb:
            wandb.init(**wandb_cfg)
            self.wandb = wandb
        else:
            self.wandb = None

        # Model to device + DDP
        self.model.to(self.device)
        if self.use_ddp:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )

        # AMP scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Logger + TB
        log_file = self.output_dir / "training.log" if self.is_main_process else None
        self.logger = setup_logger("Trainer2p5D", log_file)
        self.writer = (
            SummaryWriter(self.output_dir / "tensorboard") if self.is_main_process else None
        )

        # Metric trackers
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            mode=mode,
            verbose=self.is_main_process,
        )
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.output_dir / "checkpoints",
            max_checkpoints=max_checkpoints,
            monitor_metric=monitor_metric,
            mode=mode,
        )
        self.monitor_metric = monitor_metric
        self.monitor_mode = mode

        # State
        self.current_epoch: int = 0
        self.global_step: int = 0
        self.best_metric: Optional[float] = None

        if self.is_main_process:
            self.logger.info("=" * 70)
            self.logger.info("Trainer2p5D initialized")
            self.logger.info("=" * 70)
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"use_amp: {self.use_amp}")
            self.logger.info(f"use_ddp: {self.use_ddp}")
            if self.use_ddp:
                self.logger.info(f"world_size: {self.world_size}, local_rank: {self.local_rank}")
            self.logger.info(f"Output dir: {self.output_dir}")
            self.logger.info("=" * 70)

    # ---------- Public API ----------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> None:
        start_epoch = 0

        # Resume
        if resume_from is not None:
            if self.is_main_process:
                self.logger.info(f"Resuming from checkpoint: {resume_from}")
            ckpt = self.checkpoint_manager.load_checkpoint(
                resume_from,
                self.model,
                self.optimizer,
                self.scheduler,
                self.device,
            )
            start_epoch = ckpt["epoch"] + 1
            self.current_epoch = start_epoch
            self.global_step = start_epoch * len(train_loader)
            self.best_metric = ckpt.get("best_score", None)
            if self.is_main_process:
                self.logger.info(f"✓ Resumed from epoch {start_epoch}")
                if self.best_metric is not None:
                    self.logger.info(
                        f"  Best {self.monitor_metric} so far: {self.best_metric:.4f}"
                    )

        if self.is_main_process:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("TRAINING START")
            self.logger.info("=" * 70)
            self.logger.info(f"Total epochs: {epochs}")
            self.logger.info(f"Starting from: {start_epoch}")
            self.logger.info(f"Train batches: {len(train_loader)}")
            if val_loader is not None:
                self.logger.info(f"Val batches: {len(val_loader)}")
            self.logger.info("=" * 70)

        start_time = time.time()

        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch

            if self.use_ddp:
                dist.barrier()

            # Train
            train_metrics = self._train_epoch(train_loader)

            # Val
            val_metrics: Dict[str, float] = {}
            if val_loader is not None and ((epoch + 1) % self.val_every == 0):
                val_metrics = self._validate_epoch(val_loader)

            # Log epoch summary
            if self.is_main_process:
                self._log_epoch_summary(epoch, train_metrics, val_metrics)

            # Checkpoint
            if self.is_main_process:
                # chọn metric để monitor
                monitor_value = (
                    val_metrics.get(self.monitor_metric)
                    if val_metrics
                    else train_metrics.get("loss", 0.0)
                )

                if self.best_metric is None:
                    is_best = True
                    self.best_metric = monitor_value
                else:
                    if self.monitor_mode == "max":
                        is_best = monitor_value > self.best_metric
                    else:
                        is_best = monitor_value < self.best_metric
                    if is_best:
                        self.best_metric = monitor_value

                all_metrics = {**train_metrics, **val_metrics}
                ckpt_path = self.checkpoint_manager.save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    all_metrics,
                    is_best=is_best,
                )
                if is_best:
                    self.logger.info(
                        f" New best model! {self.monitor_metric}: {self.best_metric:.4f}"
                    )
                self.logger.info(f"Checkpoint saved: {ckpt_path}")

                if self.use_wandb:
                    log_payload = {
                        "epoch": epoch,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    }
                    for k, v in all_metrics.items():
                        log_payload[f"epoch/{k}"] = v
                    if is_best:
                        log_payload[f"epoch/best_{self.monitor_metric}"] = self.best_metric
                    self.wandb.log(log_payload)

            # Early stopping
            if val_metrics and self.is_main_process:
                monitor_value = val_metrics.get(self.monitor_metric, 0.0)
                should_stop = self.early_stopping(monitor_value, epoch)
                if should_stop:
                    self.logger.info("Early stopping triggered.")
                    break

            # Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric_for_lr = (
                        val_metrics.get(self.monitor_metric)
                        if val_metrics
                        else train_metrics.get("loss", 0.0)
                    )
                    self.scheduler.step(metric_for_lr)
                else:
                    self.scheduler.step()

        total_time = time.time() - start_time
        if self.is_main_process:
            self.logger.info("\n" + "=" * 70)
            self.logger.info("TRAINING COMPLETED")
            self.logger.info("=" * 70)
            self.logger.info(f"Total time: {total_time / 3600:.2f} hours")
            self.logger.info(f"Total epochs: {self.current_epoch + 1}")
            if self.best_metric is not None:
                self.logger.info(
                    f"Best {self.monitor_metric}: {self.best_metric:.4f}"
                )
            self.logger.info(f"Checkpoints dir: {self.checkpoint_manager.save_dir}")
            self.logger.info("=" * 70)
            if self.writer is not None:
                self.writer.close()

    # ---------- Epoch loops ----------

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        self.train_metrics.reset()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            disable=not self.is_main_process,
        )

        for batch_idx, batch in enumerate(pbar):
            batch = self._to_device(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=self.use_amp):
                loss, metrics = self._training_step(batch)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_val
                    )
                self.optimizer.step()

            self.train_metrics.update(metrics)
            self.global_step += 1

            if self.is_main_process and (batch_idx % self.log_every == 0):
                avg = self.train_metrics.average()
                lr = self.optimizer.param_groups[0]["lr"]

                if self.writer is not None:
                    self.writer.add_scalar(
                        "train/loss", avg.get("loss", 0.0), self.global_step
                    )
                    if "dice_loss" in avg:
                        self.writer.add_scalar(
                            "train/dice_loss", avg["dice_loss"], self.global_step
                        )
                    if "ce_loss" in avg:
                        self.writer.add_scalar(
                            "train/ce_loss", avg["ce_loss"], self.global_step
                        )
                    if "focal_loss" in avg:
                        self.writer.add_scalar(
                            "train/focal_loss", avg["focal_loss"], self.global_step
                        )
                    self.writer.add_scalar("train/lr", lr, self.global_step)

                if self.use_wandb:
                    log_payload = {
                        "step": self.global_step,
                        "train/lr": lr,
                    }
                    for k, v in avg.items():
                        log_payload[f"train/{k}"] = v
                    self.wandb.log(log_payload)

                pbar.set_postfix(
                    {
                        "loss": f"{avg.get('loss', 0.0):.4f}",
                        "lr": f"{lr:.6f}",
                    }
                )

        return self.train_metrics.average()

    @torch.no_grad()
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        self.val_metrics.reset()

        pbar = tqdm(
            val_loader,
            desc=f"Epoch {self.current_epoch + 1} [Val]",
            disable=not self.is_main_process,
        )

        for batch in pbar:
            batch = self._to_device(batch)
            with autocast(device_type="cuda", enabled=self.use_amp):
                metrics = self._validation_step(batch)
            self.val_metrics.update(metrics)
            avg = self.val_metrics.average()
            pbar.set_postfix(
                {k: f"{v:.4f}" for k, v in avg.items() if "val_" in k}
            )

        avg_metrics = self.val_metrics.average()

        # sync across DDP
        if self.use_ddp:
            avg_metrics = self._sync_metrics(avg_metrics)

        if self.is_main_process and self.writer is not None:
            for k, v in avg_metrics.items():
                self.writer.add_scalar(k, v, self.current_epoch)

        return avg_metrics

    # ---------- Single steps ----------

    def _training_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        center = batch["center"]      # [B,C,H,W]
        neighbors = batch["neighbors"]  # [B,K,C,H,W]
        mask = batch["mask"]          # [B,H,W]

        outputs = self.model(center, neighbors)  # Tensor hoặc list[Tensor]

        loss_dict = self.criterion(outputs, mask)
        assert "total" in loss_dict, "criterion must return dict with key 'total'"
        loss = loss_dict["total"]

        metrics: Dict[str, float] = {"loss": float(loss.item())}
        if "dice" in loss_dict:
            metrics["dice_loss"] = float(loss_dict["dice"].item())
        if "ce" in loss_dict:
            metrics["ce_loss"] = float(loss_dict["ce"].item())
        if "focal" in loss_dict:
            metrics["focal_loss"] = float(loss_dict["focal"].item())
        return loss, metrics

    def _validation_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        center = batch["center"]
        neighbors = batch["neighbors"]
        mask = batch["mask"]

        outputs = self.model(center, neighbors)

        loss_dict = self.criterion(outputs, mask)
        assert "total" in loss_dict

        # main output để tính Dice
        if isinstance(outputs, (list, tuple)):
            main_out = outputs[0]
        else:
            main_out = outputs

        pred = torch.argmax(main_out, dim=1)  # [B,H,W]
        dice = self._compute_dice(pred, mask)

        metrics: Dict[str, float] = {
            "val_loss": float(loss_dict["total"].item()),
            "val_dice": float(dice),
        }
        if "dice" in loss_dict:
            metrics["val_dice_loss"] = float(loss_dict["dice"].item())
        if "ce" in loss_dict:
            metrics["val_ce_loss"] = float(loss_dict["ce"].item())
        if "focal" in loss_dict:
            metrics["val_focal_loss"] = float(loss_dict["focal"].item())
        return metrics

    # ---------- Utils ----------

    def _compute_dice(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        pred:   [B,H,W] (class index)
        target: [B,H,W]
        """
        smooth = 1e-5
        num_classes = int(max(pred.max().item(), target.max().item()) + 1)
        dices: List[float] = []

        for c in range(num_classes):
            pred_c = (pred == c).float()
            target_c = (target == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            if union > 0:
                dice = (2.0 * intersection + smooth) / (union + smooth)
                dices.append(float(dice.item()))

        return float(np.mean(dices)) if len(dices) > 0 else 0.0

    def _to_device(self, batch: Union[Dict, torch.Tensor]) -> Union[Dict, torch.Tensor]:
        if isinstance(batch, dict):
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _sync_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        if not self.use_ddp:
            return metrics
        synced: Dict[str, float] = {}
        for k, v in metrics.items():
            t = torch.tensor(v, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            synced[k] = float((t / self.world_size).item())
        return synced

    def _log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"Epoch {epoch + 1} Summary")
        self.logger.info("=" * 70)
        self.logger.info("Train:")
        for k, v in train_metrics.items():
            self.logger.info(f"  {k}: {v:.4f}")
        if val_metrics:
            self.logger.info("Val:")
            for k, v in val_metrics.items():
                self.logger.info(f"  {k}: {v:.4f}")
        lr = self.optimizer.param_groups[0]["lr"]
        self.logger.info(f"LR: {lr:.6f}")
        self.logger.info("=" * 70)
