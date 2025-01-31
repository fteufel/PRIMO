import torch
import torchmetrics
from torchmetrics.functional import spearman_corrcoef

class AverageSpearmanCorrelation(torchmetrics.Metric):
    def __init__(self, mask_idx=-100, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_spearman", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_valid_samples", default=torch.tensor(0), dist_reduce_fx="sum")
        self.mask_idx = mask_idx

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Update state with predictions and targets.
        
        Args:
            preds (torch.Tensor): Predictions tensor of shape (batch_size, num_samples) or (num_samples,)
            target (torch.Tensor): Target tensor of shape (batch_size, num_samples) or (num_samples,)
        """
        assert preds.shape == target.shape, "Predictions and targets must have the same shape"
        
        # If inputs are 1D, unsqueeze to make them 2D
        if preds.dim() == 1:
            preds = preds.unsqueeze(0)
            target = target.unsqueeze(0)
        
        batch_size, num_samples = preds.shape

        for i in range(batch_size):
            # Create a mask where True indicates valid entries (not equal to mask_idx)
            valid_mask = target[i] != self.mask_idx
            valid_preds = preds[i][valid_mask]
            valid_targets = target[i][valid_mask]

            if len(valid_preds) > 1:  # Need at least 2 valid samples for correlation
                spearman = spearman_corrcoef(valid_preds, valid_targets)
                self.sum_spearman += spearman
                self.total_valid_samples += 1

    def compute(self):
        """
        Computes the average Spearman correlation over all samples.
        """
        if self.total_valid_samples == 0:
            return torch.tensor(0.0)  # or float('nan') if you prefer
        return self.sum_spearman / self.total_valid_samples