import argparse
import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from monai.data import (
    load_decathlon_datalist,
    ThreadDataLoader,
    CacheDataset,
    decollate_batch,
    set_track_meta,
)
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
    ToTensord,
    RandSpatialCropd
)
from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.utils import ensure_tuple_rep
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
import argparse
roi = 96
itera= 399
parser = argparse.ArgumentParser(description="PyTorch Training")
parser.add_argument("--date", default='08.16_', type=str, help="Training date")
parser.add_argument("--img_size", default=(roi,roi,roi), type=tuple, help="number of training epochs")
parser.add_argument("--mask_ratio", default=0.6, type=float, help="Training date")
parser.add_argument("--weight_name", default='90_swin_', type=str, help="SSL weight name")
parser.add_argument("--epochs", default=1600, type=int, help="number of training epochs")
parser.add_argument("--num_steps", default=1600*itera, type=int, help="number of training iterations")
parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
parser.add_argument("--eval_num", default=itera, type=int, help="evaluation frequency")

parser.add_argument("--qkv_bias", default=False, type=bool, help="warmup steps")
parser.add_argument("--save_attn", default=False, type=bool, help="warmup steps")

parser.add_argument("--warmup_steps", default=itera*20, type=int, help="warmup steps")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of input channels") 
parser.add_argument("--mlp_dim", default=3072, type=int, help="number of input channels")
parser.add_argument("--hidden_size", default=768, type=int, help="drop path rate")
parser.add_argument("--num_heads", default=12, type=int, help="drop path rate")
parser.add_argument("--num_workers", default=20, type=int, help="number of workers")
parser.add_argument("--num_layers", default=12, type=int, help="number of workers")
parser.add_argument("--feature_size", default=16, type=int, help="embedding size")
parser.add_argument("--dropout_path_rate", default=0.1, type=float, help="drop path rate")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="drop path rate")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--conv_block", default='conv', type=str, help="spatial dimension of input data")
parser.add_argument("--norm_name", default='instance', type=str, help="spatial dimension of input data")
parser.add_argument("--proj_type", default='conv', type=str, help="spatial dimension of input data")
parser.add_argument("--res_block", default=True, type=bool, help="spatial dimension of input data")
parser.add_argument("--pos_embed", default=True, type=bool, help="spatial dimension of input data")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")

parser.add_argument("--student_block", default=16, type=int, help="learning rate")

parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
parser.add_argument("--decay", default=5e-2, type=float, help="decay rate")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--lrdecay", default=True, help="enable learning rate decay")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")



args = parser.parse_args()

def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=1, prefilter=False)
    return img_resampled

def dice(x, y):
    intersect = np.sum(x * y)
    y_sum = np.sum(y)
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(x)
    return 2 * intersect / (x_sum + y_sum)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else self.sum

set_track_meta(True)
device = 'cuda'
num_samples = 4
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.5, 1.5, 2.0),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True)
    ]
)

if __name__ == "__main__":

    set_track_meta(True)
    datasets = r"/home/work/.medsam/dataset/BTCV/dataset_0.json"
    val_list = load_decathlon_datalist(datasets, True, "validation")
    val_ds = CacheDataset(data=val_list, transform=val_transforms, cache_num=len(val_list), cache_rate=1.0, num_workers=0)
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

    model = torch.load()
    model.eval()

    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = f"case_{i}"  
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(                                                                                 
                val_inputs, (96, 96, 96), 4, model, overlap=0.75)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            val_outputs = resample_3d(val_outputs, target_shape)
            dice_list_sub = []
            for i in range(1, 14):
                organ_Dice = dice(val_outputs == i, val_labels == i)
                dice_list_sub.append(organ_Dice)
                print(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)

        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))

