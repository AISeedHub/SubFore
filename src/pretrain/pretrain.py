import argparse
import os
from time import time
import logging
import numpy as np
import torch.nn.utils as torch_utils
import torch
import torch.distributed as dist
import torch.optim as optim
from monai.networks.nets import SwinUNETR
from torch.nn.parallel import DistributedDataParallel
from lr_scheduler import  WarmupCosineSchedule
import torch.optim 
from timm.utils import ModelEmaV2
from tqdm import tqdm
from lr_scheduler import WarmupCosineSchedule
from copy import deepcopy
from models.swin import Swin
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

args = parser.parse_args()


    

def validation(epoch_iterator_val):
    model.eval()
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_orgin = (batch["image"].cuda())
            with torch.amp.autocast('cuda'):
                loss = model(val_orgin)
            if loss.dim() != 0:
                loss = loss.mean()
                #loss = loss_function(val_outputs, val_inputs,idx)
    return loss

def train(global_step, train_loader, loss_val_best, global_step_best,model_ema):
    model.train()
    epoch_loss = 0
    step = 0

    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
        step += 1
        x = batch["image"].cuda()
        
        with torch.amp.autocast('cuda'):
            #x_mask,idx = masking(x)
            loss = model(x)
        
            #loss = loss_function(logit_map, x,idx)
        if loss.dim() != 0:
            loss = loss.mean()
        scaler.scale(loss).backward()
        torch_utils.clip_grad_norm_(model.parameters(),1e5) 
        epoch_loss += loss.item()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()
    
        torch.cuda.synchronize() 
        model_ema.update(model)
        optimizer.zero_grad()
        scheduler.step(global_step)
        epoch_iterator.set_description(  # noqa: B038
            f"Training ({global_step} / {max_iterations} Steps) (loss={loss:2.5f})"
        )

        if (global_step % eval_num == 0 and global_step != 0) or global_step == max_iterations:
            epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
            loss_val = validation(epoch_iterator_val)
            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)

            if loss_val.item() < loss_val_best:
                loss_val_best= loss_val
                global_step_best = global_step
                file_name = "11.25_" + str(global_step) + "_16_60_swin_pretrain_weight.pt"
                file_name2 = "11.25._" + str(global_step) + "_16_60_density_bestmodel.pt"
                torch.save(deepcopy(model.state_dict()), os.path.join(r'/home/work/.medsam/dataset/cvpr/save_weights',file_name))
                torch.save(model, os.path.join(r'/home/work/.medsam/dataset/cvpr/save_weights',file_name2))
                print(
                    "Model Was Saved ! Current Best Loss: {} Current Loss: {}".format(loss_val_best, loss_val)
                )

            else:
                print(
                    "Model Was Not Saved ! Current Best Loss. Dice: {} Current Avg. Loss: {}".format(
                        loss_val_best,loss_val))
        global_step += 1

    return global_step, loss_val_best, global_step_best


from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.utils import ensure_tuple_rep
import torch
import torch.nn.functional as F
if __name__ == "__main__":
    # try:
    
    torch.backends.cudnn.benchmark = True

    #torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Swin(args).to(device)

    #model = torch.nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])

    torch.backends.cudnn.benchmark = True
    
    from timm import optim, scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    #scheduler = scheduler.StepLRScheduler(optimizer, decay_t=460*100, decay_rate=1.0, warmup_t=500, warmup_lr_init=0, t_in_epochs=True )
    #scheduler = scheduler.CosineLRScheduler(optimizer, t_initial=args.warmup_steps)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)
    scaler = torch.amp.GradScaler()
    model_ema= ModelEmaV2(model,device=device)
    max_iterations = args.num_steps
    global_step = 0
    global_step_best = 0
    eval_num=args.eval_num
    epoch_loss_values = []
    metric_values = []
    #loss_val_best = 10000
    loss_val_best = torch.tensor(float('inf')).to(device)
    while global_step < max_iterations:
        global_step, loss_val_best, global_step_best = train(global_step, train_loader, loss_val_best, global_step_best,model_ema)