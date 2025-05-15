# Fine-Grained Classification of Forest from High-Resolution Aerial Imagery using Machine Learning

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Table of Contents
- [Overview](#overview)
- [Purpose](#purpose)
- [Key Features](#key-features)
- [Technical Description](#technical-description)
- [Installation](#installation)
- [Usage](#usage)
- [Licensing](#licensing)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Overview
This repository contains a machine-learning pipeline based on the **U-Net** architecture for automated detection and classification of tree canopies in high-resolution aerial imagery (25 cm / pixel RGB).  
The work was conducted as part of the Bachelor thesis **“Fine-Grained Classification of Forest from Aerial Imagery using Machine Learning”** at the Norwegian University of Science and Technology (NTNU) in collaboration with the Norwegian Military Academy.

## Purpose
Public datasets such as **AR5** and **FKB Arealdekke** suffer from coarse resolution and slow update cycles, limiting their utility for tasks like military mobility analysis.  
Our model delivers finer spatial detail, faster update potential, and improved accuracy for tree-cover mapping.

## Key Features
- **U-Net** semantic-segmentation architecture  
- **High-quality training set** built from RGB orthophotos and digital-elevation models  
- **Automated workflow**: preprocessing → training → raster prediction  
- Accuracy **79.4 %**, F1-score **78.4 %** on test data  
- Outperforms existing datasets, especially in sparsely vegetated or mixed-cover areas  

## Technical Description
- **Data Processing** → `dataloader.py`, `Model_To_Raster.py`  
- **Model & Training** → `model.py`, `losses.py`, `train.py`  
- **Evaluation & Visualisation** → `visualization.py`, `test_model.py`  
- **Utilities / examples** → `ReLU.py`, `test_raster_opening.py`  

---

## Installation
We recommend a Conda setup.

### 1. Create the environment
```bash
conda env create -f environment.yml
```

### 2. Activate it
```bash
conda activate treecanopy
```

If you see a *“missing `pytorch-cpu`”* warning, you can safely ignore it; the environment installs the GPU-enabled build (`pytorch-cuda`) instead.

#### Main dependencies
- Python 3.11  
- PyTorch 2.2.2 (GPU, CUDA 12.1)  
- torchvision 0.17.2  
- numpy, rasterio, matplotlib, seaborn, scikit-learn, tqdm  
- opencv-python-headless  

> **Note:** If your NVIDIA driver < 528, edit `environment.yml` and change `pytorch-cuda` to `11.8`.

---

## Usage

### Training the model
```bash
python train.py --data /path/to/dataset --num_epochs 50 --batch 8 --loss focalloss
```

### Running inference
```bash
python Model_To_Raster.py --data_root /path/to/data --model_path /path/to/model.pth --batch_size 4
```

### Visualisation
```bash
python visualization.py --data_root /path/to/data --model_path /path/to/model.pth
```

---

## Licensing
This project is released under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for full terms.

---

## Acknowledgements
This repository includes code adapted from prior work distributed under [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html).

**Supervisors:** Bjørn Wangensteen · Sverre Stikbakke · Eina Bergem Jørgensen

---

## Contact
- **Authors:** Aslak Brynsrud · Hans Olav Lien Johansen  
- **Institution:** NTNU, Department of Manufacturing and Civil Engineering  
- **Date of Submission:** May 2024
