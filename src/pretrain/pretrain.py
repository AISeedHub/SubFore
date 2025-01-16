import os
import torch
from tqdm import tqdm
from copy import deepcopy
import torch.nn.utils as torch_utils
import torch
from lr_scheduler import  WarmupCosineSchedule
import torch.optim 
import argparse
from tqdm import tqdm
from timm import scheduler
from timm.utils import ModelEmaV2
from torch.amp import GradScaler, autocast
from models.swin import Swin
from datasets import train_loader,val_loader
from lr_scheduler import WarmupCosineSchedule

roi = 96
itera= 399
parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--date", default='08.16_', type=str, help="Training date")
parser.add_argument("--mask_ratio", default=0.6, type=float, help="Training date")
parser.add_argument("--weight_name", default='90_swin_', type=str, help="SSL weight name")
parser.add_argument("--epochs", default=1600, type=int, help="number of training epochs")
parser.add_argument("--num_steps", default=1600*itera, type=int, help="number of training iterations")
parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
parser.add_argument("--eval_num", default=itera, type=int, help="evaluation frequency")
parser.add_argument("--warmup_steps", default=itera*20, type=int, help="warmup steps")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--num_workers", default=92, type=int, help="number of workers")
parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
parser.add_argument("--dropout_path_rate", default=0.1, type=float, help="drop path rate")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")
parser.add_argument("--subvolume_size", default=16, type=int, help="size of subvolume")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--lr", default=8e-4, type=float, help="learning rate")
parser.add_argument("--decay", default=5e-2, type=float, help="decay rate")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--lrdecay", default=True, help="enable learning rate decay")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")
parser.add_argument("--luna_path", default='./luna', type=str, help="LUNA 16 dataset path")
parser.add_argument("--btcv_path", default='./btcv', type=str, help="BTCV dataset path")
parser.add_argument("--covid_path", default='./covid19', type=str, help="TCIA Covid 19 dataset path")
args = parser.parse_args()


def validation(epoch_iterator_val, model):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_origin = batch["image"].cuda()
            with autocast(device_type='cuda'):
                loss = model(val_origin)
            if loss.dim() != 0:
                loss = loss.mean()
    return loss


def train(global_step, train_loader, loss_val_best, global_step_best, model_ema, model, optimizer, scheduler, scaler, max_iterations, eval_num, val_loader, args):
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
        torch_utils.clip_grad_norm_(model.parameters(), 1e5)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step(global_step)

        epoch_loss += loss.item()
        epoch_iterator.set_description(
            f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})"
        )

        model_ema.update(model)
        torch.cuda.synchronize()

        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate", dynamic_ncols=True)
            loss_val = validation(epoch_iterator_val, model)

            epoch_loss /= (step + 1)

            if loss_val.item() < loss_val_best:
                loss_val_best = loss_val.item()
                global_step_best = global_step

                file_name = f"{global_step}_16_60_swin_pretrain_weight.pt"
                file_path = os.path.join(args.save_path, file_name)

                torch.save(deepcopy(model.state_dict()), file_path)

                print(f"Model Was Saved! Current Best Loss: {loss_val_best:.4f}, Current Loss: {loss_val.item():.4f}")
            else:
                print(f"Model Was Not Saved! Current Best Loss: {loss_val_best:.4f}, Current Avg. Loss: {loss_val.item():.4f}")

        global_step += 1

    return global_step, loss_val_best, global_step_best


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Swin(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)
    scaler = GradScaler()
    model_ema = ModelEmaV2(model, device=device)

    max_iterations = args.num_steps
    global_step = 0
    global_step_best = 0
    eval_num = args.eval_num
    loss_val_best = float('inf')

    while global_step < max_iterations:
        global_step, loss_val_best, global_step_best = train(
            global_step, train_loader, loss_val_best, global_step_best,
            model_ema, model, optimizer, scheduler, scaler,
            max_iterations, eval_num, val_loader, args
        )
