import torch
import torch.nn as nn
from monai.networks.blocks import UnetrBasicBlock, UnetrUpBlock, UnetOutBlock
from monai.utils import ensure_tuple_rep
from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from Pretraining.loss import Reconstruction_loss
from Pretraining.pretrain import args

class Swin(nn.Module):
    def __init__(self, args):
        super(Swin, self).__init__()

        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)

        self.swinViT = SwinViT(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
            use_v2=True,
        )

        norm_name = 'instance'

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size,
            out_channels=args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=2 * args.feature_size,
            out_channels=2 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=4 * args.feature_size,
            out_channels=4 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=16 * args.feature_size,
            out_channels=16 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=16 * args.feature_size,
            out_channels=8 * args.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=8 * args.feature_size,
            out_channels=4 * args.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=4 * args.feature_size,
            out_channels=2 * args.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=2 * args.feature_size,
            out_channels=args.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size,
            out_channels=args.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size,
            out_channels=args.out_channels,
        )

        self.recon_loss = Reconstruction_loss()

    def masking(self, x_asd, masking_ratio, block_size):
        """
        Applies masking to the input tensor with Gaussian noise.
        """
        batch_ct_array = x_asd.clone()
        num_samples, c, z, y, x = batch_ct_array.shape

        pad_z = (block_size - (z % block_size)) % block_size
        pad_y = (block_size - (y % block_size)) % block_size
        pad_x = (block_size - (x % block_size)) % block_size

        batch_ct_array = torch.nn.functional.pad(
            batch_ct_array, (0, pad_x, 0, pad_y, 0, pad_z)
        )

        _, _, z_padded, y_padded, x_padded = batch_ct_array.shape
        num_blocks_z = z_padded // block_size
        num_blocks_y = y_padded // block_size
        num_blocks_x = x_padded // block_size

        batch_ct_array_blocks = batch_ct_array.view(
            num_samples, c, num_blocks_z, block_size, num_blocks_y, block_size, num_blocks_x, block_size
        ).permute(0, 1, 2, 4, 6, 3, 5, 7)

        block_means = batch_ct_array_blocks.mean(dim=(5, 6, 7))
        condition = (block_means >= 0.1) & (block_means <= 1.0)
        random_probs = torch.rand_like(block_means)
        mask_blocks = condition & (random_probs < masking_ratio)

        masked_block_indices = torch.nonzero(mask_blocks)
        mask_blocks = mask_blocks.unsqueeze(-1).expand(-1, -1, -1, -1, block_size, block_size, block_size)
        mask = mask_blocks.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous().view(num_samples, c, z_padded, y_padded, x_padded)

        noise = torch.normal(mean=0.0, std=1.0, size=batch_ct_array.shape, device=batch_ct_array.device)
        masked_array = torch.where(mask, noise, batch_ct_array)

        return masked_array, masked_block_indices

    def forward(self, x_in):
        mask_x, mask_x_index = self.masking(
            x_asd=x_in, masking_ratio=args.mask_ratio, block_size=args.subvolume_size
        )

        hidden_states_out = self.swinViT(mask_x)

        enc0 = self.encoder1(mask_x)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        logit = self.out(out)
        total_loss = self.recon_loss(logit, x_in, mask_x_index)

        return total_loss