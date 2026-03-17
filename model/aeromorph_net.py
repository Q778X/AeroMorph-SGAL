import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import R2Score, MeanAbsoluteError, MeanSquaredError

from .point_transformer import PointTransformerEncoder
from .implicit_sdf import ImplicitSDF

try:
    from utils.loss import AeroMorphLoss
    _AERO_LOSS_AVAILABLE = True
except ImportError:
    _AERO_LOSS_AVAILABLE = False
    print("Warning: utils/loss.py not found, physics loss disabled.")

try:
    from utils.visualize import visualize_pressure_field
    _VIS_AVAILABLE = True
except ImportError:
    _VIS_AVAILABLE = False


class AeroMorphNetLightning(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        self.encoder = PointTransformerEncoder(cfg)

        self.sdf = ImplicitSDF(cfg)

        fusion_dim = cfg["model"]["fusion"]["hidden_dim"]
        control_flat_dim = cfg["model"]["implicit_sdf"]["control_points"] * 3

        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim + control_flat_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(cfg["model"]["fusion"].get("dropout", 0.1)),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, 1)
        )

        self.use_aero_loss = cfg["train"].get("use_aero_loss", False)
        if self.use_aero_loss and _AERO_LOSS_AVAILABLE:
            self.aero_loss_fn = AeroMorphLoss(cfg)
        elif self.use_aero_loss:
            print("Warning: use_aero_loss=True but module is missing. Disabled automatically.")
            self.use_aero_loss = False

        self.val_r2 = R2Score()
        self.val_mae = MeanAbsoluteError()
        self.val_mse = MeanSquaredError()

    def forward(self, points, return_grad=False):
        B = points.shape[0]

        explicit_feat = self.encoder(points)

        control_flat = self.sdf.control_points.view(1, -1).repeat(B, 1)

        grad = None
        if return_grad or (self.training and self.use_aero_loss):
            grad = self.sdf.get_gradient(points)

        fused = torch.cat([explicit_feat, control_flat], dim=1)
        cd_pred = self.fusion(fused).squeeze(-1)

        return cd_pred, grad

    def training_step(self, batch, batch_idx):
        points = batch["points"]
        cd_true = batch["cd"]

        cd_pred, grad = self(points)

        loss_cd = F.mse_loss(cd_pred, cd_true)

        self.log("train_mse", loss_cd, prog_bar=True, on_step=True)

        if self.use_aero_loss and grad is not None:
            total_loss, loss_dict = self.aero_loss_fn(cd_pred, cd_true, grad, points)

            for k, v in loss_dict.items():
                self.log(f"train_{k}", v, on_step=True, on_epoch=False, prog_bar=(k=="loss_physics"))

            return total_loss

        return loss_cd

    def validation_step(self, batch, batch_idx):
        points = batch["points"]
        cd_true = batch["cd"]

        cd_pred, _ = self(points, return_grad=False)

        self.val_r2.update(cd_pred, cd_true)
        self.val_mae.update(cd_pred, cd_true)
        self.val_mse.update(cd_pred, cd_true)

    def on_validation_epoch_end(self):
        r2 = self.val_r2.compute()
        mae = self.val_mae.compute()
        mse = self.val_mse.compute()

        self.log("val_r2", r2, prog_bar=True)
        self.log("val_mae", mae, prog_bar=True)
        self.log("val_mse", mse, prog_bar=True)

        self.val_r2.reset()
        self.val_mae.reset()
        self.val_mse.reset()

        print(f"\n[Epoch {self.current_epoch} Val] R2: {r2:.4f} | MAE: {mae:.4f}")

    def test_step(self, batch, batch_idx):
        points = batch["points"]
        cd_true = batch["cd"]
        name = batch["name"][0] if isinstance(batch["name"], (list, tuple)) else batch["name"]

        if points.dim() == 2:
            points = points.unsqueeze(0)

        cd_pred, _ = self(points, return_grad=False)
        cd_pred_val = cd_pred.item()
        cd_true_val = cd_true.item() if cd_true is not None else 0.0

        if _VIS_AVAILABLE and self.cfg["inference"].get("output_vis", False):
            visualize_pressure_field(
                points[0].cpu(),
                cd_pred=round(cd_pred_val, 5),
                cd_true=round(cd_true_val, 5),
                name=name,
                save_path="outputs/vis_test"
            )

        self.log("test_mae", F.l1_loss(cd_pred, cd_true))
        print(f"[TEST] {name} -> Pred: {cd_pred_val:.5f} | True: {cd_true_val:.5f} | Diff: {abs(cd_pred_val - cd_true_val):.5f}")

        return {"pred": cd_pred_val, "true": cd_true_val}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg["train"]["lr"],
            weight_decay=self.cfg["train"].get("weight_decay", 1e-4)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg["train"]["epochs"],
            eta_min=1e-6
        )
        return [optimizer], [scheduler]