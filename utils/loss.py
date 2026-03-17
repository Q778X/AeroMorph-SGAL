import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import grad_norm

class AeroMorphLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mse = nn.MSELoss()

        aero_cfg = cfg.get("train", {}).get("aero_loss", {})
        self.velocity_dir = torch.tensor(
            aero_cfg.get("velocity_dir", [-1.0, 0.0, 0.0]),
            dtype=torch.float32
        )
        self.front_axis = aero_cfg.get("front_axis", 0)
        self.front_threshold = aero_cfg.get("front_threshold", 0.1)
        self.grad_weight = aero_cfg.get("grad_weight", 0.1)
        self.smooth_weight = aero_cfg.get("smooth_weight", 0.01)

    def forward(self, cd_pred, cd_true, grad_field, points):
        loss_cd = self.mse(cd_pred, cd_true)

        B, N, _ = grad_field.shape
        device = grad_field.device

        velocity_dir = self.velocity_dir.type_as(grad_field).view(1, 1, 3).expand(B, N, -1)
        velocity_dir = F.normalize(velocity_dir, dim=-1)

        grad_norm_vec = F.normalize(grad_field, dim=-1)

        cos_angle = (grad_norm_vec * velocity_dir).sum(dim=-1)

        coords = points[:, :, self.front_axis]
        min_val = coords.min(dim=1, keepdim=True)[0]
        max_val = coords.max(dim=1, keepdim=True)[0]

        length = max_val - min_val
        threshold_val = min_val + self.front_threshold * length

        front_mask = coords < threshold_val

        if front_mask.sum() == 0:
            loss_physics = torch.tensor(0.0, device=device)
        else:
            loss_physics = F.relu(cos_angle[front_mask]).mean()

        grad_magnitude = grad_field.norm(dim=-1)
        loss_smooth = torch.var(grad_magnitude, dim=1).mean()

        total_loss = loss_cd + self.grad_weight * loss_physics + self.smooth_weight * loss_smooth

        return total_loss, {
            "loss_cd": loss_cd.item(),
            "loss_physics": loss_physics.item(),
            "loss_smooth": loss_smooth.item()
        }