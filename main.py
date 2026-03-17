import os
import torch
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()

project_root = Path(__file__).parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import yaml
import pandas as pd

from data.dataset import DrivAerDataModuleCSV as DrivAerDataModule
from model.aeromorph_net import AeroMorphNetLightning

torch.set_num_threads(8)
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

try:
    from utils.loss import AeroMorphLoss
    from utils.visualize import visualize_pressure_field
    print("Successfully imported custom utils modules.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

def main():
    cfg = load_config()

    datamodule = DrivAerDataModule(cfg)

    if cfg["mode"] in ["eval", "inference"] and cfg["inference"].get("model_path"):
        ckpt_path = cfg["inference"]["model_path"]
        print(f"Loading pre-trained model weights: {ckpt_path}")
        model = AeroMorphNetLightning.load_from_checkpoint(ckpt_path, cfg=cfg)
    else:
        print("Initializing model from scratch.")
        model = AeroMorphNetLightning(cfg)

    logger = CSVLogger(
        save_dir="outputs/logs_csv",
        name="aeromorph",
        version=f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )
    print(f"Logs will be saved at: {logger.log_dir}")

    checkpoint = ModelCheckpoint(
        dirpath="outputs/checkpoints",
        filename="best-{epoch:03d}-{val_mae:.5f}",
        monitor="val_mae",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    train_cfg = cfg["train"]
    trainer = Trainer(
        max_epochs=train_cfg["epochs"],
        accelerator=train_cfg.get("accelerator", "gpu"),
        devices=train_cfg.get("devices", 1),
        strategy=train_cfg.get("strategy", "auto"),
        precision=train_cfg.get("precision", "bf16-mixed"),
        accumulate_grad_batches=train_cfg.get("accumulate_grad_batches", 1),
        gradient_clip_val=train_cfg.get("gradient_clip_val", 1.0),
        log_every_n_steps=train_cfg.get("log_every_n_steps", 20),
        num_sanity_val_steps=train_cfg.get("num_sanity_val_steps", 0),
        deterministic=train_cfg.get("deterministic", False),
        benchmark=train_cfg.get("benchmark", True),
        logger=logger,
        callbacks=[checkpoint, lr_monitor],
        enable_progress_bar=True,
    )

    print(f"Starting {cfg['mode']} | Precision: {train_cfg.get('precision')} | BatchSize: {cfg['data']['batch_size']}")

    if cfg["mode"] == "train":
        trainer.fit(model, datamodule)

        metrics_path = os.path.join(logger.log_dir, "metrics.csv")
        if os.path.exists(metrics_path):
            df = pd.read_csv(metrics_path)
            clean_df = df.drop_duplicates(subset=["epoch"], keep="last")
            clean_path = "outputs/logs_csv/training_history_clean.csv"
            os.makedirs(os.path.dirname(clean_path), exist_ok=True)
            clean_df.to_csv(clean_path, index=False)
            print(f"\nTraining complete! Clean logs saved -> {clean_path}")
            print("Summary of the last 10 epochs:")
            target_cols = ["epoch", "train_mse", "val_mae", "val_mse", "val_r2"]
            existing_cols = [c for c in target_cols if c in clean_df.columns]
            if existing_cols:
                print(clean_df[existing_cols].tail(10).to_string(index=False))
            else:
                print("Warning: Expected log columns not found, please check metrics.csv.")

    elif cfg["mode"] == "eval":
        trainer.validate(model, datamodule)
    elif cfg["mode"] == "inference":
        trainer.test(model, datamodule)
    elif cfg["mode"] == "visualize":
        model.visualize_samples(datamodule, num_samples=10)

if __name__ == "__main__":
    main()