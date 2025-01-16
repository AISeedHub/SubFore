import torch.nn as nn
from Pretraining.pretrain import args

class ReconstructionLoss(nn.Module):
    def __init__(self, block_size):
        super(ReconstructionLoss, self).__init__()
        self.criterion_L1 = nn.L1Loss(reduction='mean')
        self.block_size = block_size

    def calculate_loss(self, pred, gt):
        """Calculate L1 loss between predicted and ground truth tensors."""
        return self.criterion_L1(gt, pred)

    def forward(self, masked_tensor, original_tensor, masked_indices):
        """
        Compute reconstruction loss for the masked regions.

        Args:
            masked_tensor: Tensor with masked regions.
            original_tensor: Original tensor before masking.
            masked_indices: List of indices for masked regions.

        Returns:
            Reconstruction loss averaged over all masked regions.
        """
        if len(masked_indices) == 0:
            return 0.0

        total_loss = 0.0
        for idx in masked_indices:
            # Unpack block indices
            sample_idx, _, block_z, block_y, block_x = idx

            # Calculate block coordinates
            z = block_z * self.block_size
            y = block_y * self.block_size
            x = block_x * self.block_size

            # Extract blocks from tensors
            original_block = original_tensor[
                sample_idx, :, z:z + self.block_size, y:y + self.block_size, x:x + self.block_size
            ]
            masked_block = masked_tensor[
                sample_idx, :, z:z + self.block_size, y:y + self.block_size, x:x + self.block_size
            ]

            # Accumulate loss for the current block
            total_loss += self.calculate_loss(masked_block, original_block)

        return total_loss / len(masked_indices)
