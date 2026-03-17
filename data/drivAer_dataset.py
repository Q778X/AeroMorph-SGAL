import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
import numpy as np

class DrivAerDatasetWithCSV(Dataset):
    def __init__(self, split="train", root_dir=None, csv_path=None, num_points=2048):
        self.split = split
        self.num_points = num_points
        self.root_dir = Path(root_dir)

        txt_path = Path(f"data/subsets/{split}.txt")

        if not txt_path.exists():
            raise FileNotFoundError(f"Split file not found: {txt_path}")

        with open(txt_path, "r", encoding="utf-8") as f:
            design_ids = [line.strip() for line in f if line.strip()]

        if not Path(csv_path).exists():
            raise FileNotFoundError(f"CSV label file not found: {csv_path}")
        df = pd.read_csv(csv_path)

        df.columns = df.columns.str.strip()

        id_col = df.columns[0]
        cd_candidates = ["cd", "Cd", "CD", "cd_value", "Drag", "drag_coeff"]
        cd_col = next((c for c in cd_candidates if c in df.columns), None)

        if cd_col is None:
            raise ValueError(f"Cannot find Cd column in CSV. Available columns: {list(df.columns)}")

        df = df.set_index(id_col)
        self.cd_map = df[cd_col].to_dict()

        self.samples = []
        for design_id in design_ids:
            pt_path = self.root_dir / f"{design_id}.pt"
            if not pt_path.exists():
                continue
            if design_id in self.cd_map or split == "test":
                self.samples.append((design_id, pt_path))

        print(f"[{split.upper()}] Successfully loaded {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        design_id, pt_path = self.samples[idx]

        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
        except Exception:
            data = torch.load(pt_path, map_location="cpu")

        if isinstance(data, dict):
            points = data.get("points") or data.get("point_cloud")
        else:
            points = data[:, :3]

        points = points.float()

        centroid = torch.mean(points, dim=0)
        points = points - centroid
        m = torch.max(torch.sqrt(torch.sum(points ** 2, dim=1)))
        points = points / (m + 1e-6)

        orig_num = points.shape[0]
        if orig_num >= self.num_points:
            idx = torch.randperm(orig_num)[:self.num_points]
            points = points[idx]
        else:
            choice = torch.randint(0, orig_num, (self.num_points,))
            points = points[choice]

        cd_value = self.cd_map.get(design_id, 0.0)
        cd = torch.tensor(cd_value, dtype=torch.float32)

        return {
            "points": points,
            "cd": cd,
            "name": design_id
        }

class DrivAerDataModuleCSV(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def train_dataloader(self):
        return DataLoader(
            DrivAerDatasetWithCSV("train", **self._get_args()),
            batch_size=self.cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=self.cfg["data"].get("num_workers", 4),
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            DrivAerDatasetWithCSV("val", **self._get_args()),
            batch_size=self.cfg["data"]["batch_size"],
            shuffle=False,
            num_workers=self.cfg["data"].get("num_workers", 4),
            pin_memory=True,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            DrivAerDatasetWithCSV("test", **self._get_args()),
            batch_size=1,
            shuffle=False
        )

    def _get_args(self):
        return {
            "root_dir": self.cfg["data"]["root_dir"],
            "csv_path": self.cfg["data"]["csv_path"],
            "num_points": self.cfg["data"]["num_points"]
        }