# AeroMorph: Physics-Aware 3D Point Cloud Network for Vehicle Aerodynamics

This repository contains the official PyTorch implementation for the paper:  
**"Self-Supervised Physical Constraint Learning for 3D Vehicle Aerodynamics using Implicit-Explicit Fusion"** (Currently under peer review).

## 📌 Overview
[cite_start]AeroMorph is a physics-aware hybrid framework designed for rapid aerodynamic drag prediction ($C_d$) and flow topology reconstruction directly from raw 3D point clouds[cite: 6, 47]. [cite_start]By integrating an explicit Point Transformer stream [cite: 9, 188] [cite_start]with an implicit neural representation [cite: 9, 237][cite_start], the model captures both high-frequency geometric details and global potential field consistency[cite: 50, 85].

### Key Contributions:
* [cite_start]**CAWS Strategy**: A Curvature-Aerodynamic Weighted Sampling method that preserves critical boundary layer features[cite: 7, 48, 49].
* [cite_start]**SGAL Loss**: An unsupervised Streamline Gradient Alignment Loss that enforces anti-penetration physical constraints at the windward stagnation zone[cite: 10, 11, 51, 52].
* [cite_start]**High Efficiency**: Achieves a stable MAE of $0.0094$ on the DrivAerNet++ dataset with millisecond-level inference speed[cite: 12, 13, 14].

## 📂 Repository Structure
```text
AeroMorph_SGAL/
├── data/
│   ├── dataset.py            # Data loading and normalization
│   └── subsets/              # Train/Val/Test split IDs (.txt)
├── model/
│   ├── __init__.py
│   ├── aeromorph_net.py      # Main LightningModule [cite: 351]
│   ├── implicit_sdf.py       # Implicit field reconstruction [cite: 102]
│   └── point_transformer.py  # Explicit feature encoder [cite: 91]
├── utils/
│   ├──loss.py               # SGAL and multi-objective loss [cite: 259, 294]
│
├── config.yaml               # Global hyperparameters [cite: 360]
├── main.py                   # Training and evaluation entry [cite: 351]
├── requirements.txt
└── README.md