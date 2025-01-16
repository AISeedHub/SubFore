import argparse
import os
from copy import deepcopy
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils
import torch.optim as optim
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from timm.utils import accuracy, ModelEmaV2
from tqdm import tqdm

from monai.data import (
    CacheDataset, DataLoader, Dataset, PersistentDataset, decollate_batch,
    load_decathlon_datalist, set_track_meta
)
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.nets import SwinUNETR
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.transforms import (
    AsDiscrete, Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped, LoadImaged,
    NormalizeIntensityd, Orientationd, RandCropByPosNegLabeld, RandSpatialCropd,
    RandSpatialCropSamplesd, ScaleIntensityRanged, SpatialPadd, Spacingd, ToTensord
)
from monai.utils import ensure_tuple_rep

from lr_scheduler import LinearWarmupCosineAnnealingLR, WarmupCosineSchedule
from loss.loss import DC_0
from utils import masking

# Argument Parser Setup
parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--date", default='08.16_', type=str, help="Training date")
parser.add_argument("--mask_ratio", default=0.6, type=float, help="Mask ratio")
parser.add_argument("--weight_name", default='90_swin_', type=str, help="SSL weight name")
parser.add_argument("--epochs", default=1600, type=int, help="Number of training epochs")
parser.add_argument("--num_steps", default=1600 * 399, type=int, help="Number of training iterations")
parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
parser.add_argument("--eval_num", default=399, type=int, help="Evaluation frequency")
parser.add_argument("--warmup_steps", default=399 * 20, type=int, help="Warmup steps")
parser.add_argument("--in_channels", default=1, type=int, help="Input channels")
parser.add_argument("--out_channels", default=1, type=int, help="Output channels")
parser.add_argument("--num_workers", default=92, type=int, help="Number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="Feature size")
parser.add_argument("--dropout_path_rate", default=0.1, type=float, help="Drop path rate")
parser.add_argument("--use_checkpoint", action='store_true', help="Use gradient checkpointing")
parser.add_argument("--spatial_dims", default=3, type=int, help="Spatial dimensions")
parser.add_argument("--a_min", default=-175.0, type=float, help="Minimum intensity scale")
parser.add_argument("--a_max", default=250.0, type=float, help="Maximum intensity scale")
parser.add_argument("--b_min", default=0.0, type=float, help="Normalized minimum")
parser.add_argument("--b_max", default=1.0, type=float, help="Normalized maximum")
parser.add_argument("--space_x", default=1.5, type=float, help="Spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="Spacing in y direction")
parser.add_argument("--space_z", default=1.5, type=float, help="Spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="ROI size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="ROI size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="ROI size in z direction")
parser.add_argument("--student_block", default=32, type=int, help="Student block size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="Sliding window batch size")
parser.add_argument("--lr", default=8e-4, type=float, help="Learning rate")
parser.add_argument("--decay", default=5e-2, type=float, help="Weight decay")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum")
parser.add_argument("--lrdecay", action='store_true', help="Enable learning rate decay")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Maximum gradient norm")
args = parser.parse_args()

# Preprocessing and Dataset Preparation
train_transforms = Compose([
    LoadImaged(keys=["image"], image_only=True),
    EnsureChannelFirstd(keys=["image"]),
    Spacingd(keys=["image"], pixdim=(args.space_x, args.space_y, args.space_z), mode="bilinear"),
    Orientationd(keys=["image"], axcodes="RAS"),
    ScaleIntensityRanged(
        keys=["image"], a_min=args.a_min, a_max=args.a_max,
        b_min=args.b_min, b_max=args.b_max, clip=True
    ),
    SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
    RandSpatialCropd(
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        keys=["image"], random_size=False, random_center=True
    ),
    ToTensord(keys=["image"]),
    EnsureTyped(keys=["image"], device='cpu', track_meta=False),
])

# Dataset loading and validation
# Please add appropriate dataset loading logic here based on your setup.

def validation(epoch_iterator_val, model):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_orgin = batch["image"].cuda()
            with autocast(device_type='cuda'):
                loss = model(val_orgin)
            if loss.dim() != 0:
                loss = loss.mean()
    return loss

def train(global_step, train_loader, loss_val_best, global_step_best, model_ema, model, optimizer, scheduler, scaler, max_iterations, eval_num, val_loader):
    model.train()
    epoch_loss = 0

    epoch_iterator = tqdm(train_loader, desc="Training", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        x = batch["image"].cuda()

        with autocast(device_type='cuda'):
            loss = model(x)

        if loss.dim() != 0:
            loss = loss.mean()

        scaler.scale(loss).backward()
        torch_utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step(global_step)

        epoch_loss += loss.item()
        epoch_iterator.set_postfix(loss=loss.item())

        if global_step % eval_num == 0:
            val_loss = validation(tqdm(val_loader, desc="Validation", dynamic_ncols=True), model)
            if val_loss.item() < loss_val_best:
                loss_val_best = val_loss.item()
                global_step_best = global_step
                print(f"New best model saved at step {global_step}")

        global_step += 1

    return global_step, loss_val_best, global_step_best

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SwinViT(args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = WarmupCosineSchedule(
        optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps
    )
    scaler = GradScaler()
    model_ema = ModelEmaV2(model, device=device)

    global_step, loss_val_best, global_step_best = 0, float('inf'), 0

    # Add your dataset and dataloader initialization here.

    while global_step < args.num_steps:
        global_step, loss_val_best, global_step_best = train(
            global_step, train_loader, loss_val_best, global_step_best,
            model_ema, model, optimizer, scheduler, scaler,
            args.num_steps, args.eval_num, val_loader
        )
