# AeroMorph: Physics-Aware 3D Point Cloud Network for Vehicle Aerodynamics

This repository contains the official PyTorch implementation for the paper:  
**"Self-Supervised Physical Constraint Learning for 3D Vehicle Aerodynamics using Implicit-Explicit Fusion"** (Currently under peer review).

---

## 📌 Overview
**AeroMorph** is a physics-aware hybrid framework designed for rapid aerodynamic drag prediction ($C_d$) and flow topology reconstruction directly from raw 3D point clouds. By integrating an explicit Point Transformer stream with an implicit neural representation, the model captures both high-frequency geometric details and global potential field consistency.

### 🌟 Key Contributions
* **CAWS Strategy**: A Curvature-Aerodynamic Weighted Sampling method that preserves critical boundary layer features.
* **SGAL Loss**: An unsupervised *Streamline Gradient Alignment Loss* that enforces anti-penetration physical constraints at the windward stagnation zone.
* **High Efficiency**: Achieves a stable MAE of **0.0094** on the DrivAerNet++ dataset with millisecond-level inference speed.

---

## 📂 Repository Structure
```text
AeroMorph_SGAL/
├── data/
│   ├── dataset.py            # Data loading and normalization
│   └── subsets/              # Train/Val/Test split IDs (.txt)
├── model/
│   ├── aeromorph_net.py      # Main LightningModule integration
│   ├── implicit_sdf.py       # Implicit neural field reconstruction
│   └── point_transformer.py  # Explicit feature encoder
├── utils/
│   ├── loss.py               # SGAL and multi-objective loss
│   
├── config.yaml               # Global hyperparameters
├── main.py                   # Training and evaluation entry
├── requirements.txt          # Environment dependencies
└── README.md                 # Project documentation
```
## ⚙️ Setup and Installation
Recommended Environment: Python 3.9+, CUDA 11.8+.
```bash
# Create a virtual environment
conda create -n aeromorph python=3.9 -y
conda activate aeromorph

# Install dependencies
pip install -r requirements.txt

# Note: Install the appropriate PyTorch version for your CUDA environment
# e.g., pip install torch torchvision --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
```
## Dataset Preparation
The framework is evaluated on the DrivAerNet++ dataset.
Download the raw mesh data from the DrivAerNet++ official repository.
Preprocess the data into point clouds (4,096 points) using the CAWS strategy.
Organize the directory as follows:
Plaintext
data/
├── aero_metadata.csv      # Drag coefficient labels
├── subsets/               # train.txt, val.txt, test.txt
└── point_clouds/          # Preprocessed .pt files
## 🚀 Usage
1. TrainingThe training process utilizes BF16 mixed-precision and gradient accumulation to simulate a global batch size of 32.
```bash
python main.py --mode train --config config.yaml
```
2. Evaluation / InferenceTo verify the $R^2$ and $MAE$ metrics on the independent test set:
```bash
python main.py --mode inference --config config.yaml
```
## 📈 Results
Our model achieves state-of-the-art performance in vehicle drag prediction:
MAE: 0.0094 ± 0.0004
R² Score: 0.8962 ± 0.0080
Inference Speed: ~15ms per geometry on a single NVIDIA GPU.
## 📜 Citation (Anonymous)

@article{aeromorph2026,
  title={Self-Supervised Physical Constraint Learning for 3D Vehicle Aerodynamics using Implicit-Explicit Fusion},
  author={Anonymous Authors},
  journal={Submitted for Peer Review},
  year={2026}
}
## 🛡️ Data Availability
The source code is provided for peer-review purposes only. For double-blind review, all identifiable information has been anonymized.
