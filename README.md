# Fine-Grained Classification of Forest from High-Resolution Aerial Imagery using Machine Learning

## Overview

This repository contains a machine learning model based on the U-Net architecture for the automated detection and classification of tree canopies using high-resolution aerial imagery. The model has been specifically trained on RGB images with a resolution of 25 cm per pixel, enabling detailed and up-to-date forest cover mapping compared to traditional datasets.

This project was part of a Bachelor thesis titled **"Fine-Grained Classification of Forest from Aerial Imagery using Machine Learning"**, conducted at the Norwegian University of Science and Technology (NTNU), in collaboration with the Norwegian Military Academy.

## Purpose

Current publicly available datasets such as AR5 and FKB Arealdekke exhibit limitations in spatial resolution and update frequency, affecting their utility in applications like military mobility analysis. This project addresses these limitations by developing a robust method for classifying tree cover with higher precision and quicker update capabilities.

## Key Features

* **U-Net architecture** tailored for semantic segmentation of aerial imagery.
* **High-quality training dataset** created using RGB aerial photographs and digital elevation models (DEM).
* **Automated workflow** for generating accurate raster maps from model predictions.
* Achieved an accuracy of **79.4%** and an **F1-score of 78.4%** during evaluation.
* Demonstrates significant improvements over existing datasets, especially in areas of sparse vegetation or mixed land covers.

## Technical Description

The repository includes:

* **Data Processing Scripts**: Tools for preprocessing, data augmentation, and raster conversion (`dataloader.py`, `Model_To_Raster.py`).
* **Model Definition and Training**: U-Net implementation (`model.py`), custom loss functions (`losses.py`), and training pipeline (`train.py`).
* **Visualization and Evaluation**: Evaluation scripts generating confusion matrices and overlays for intuitive analysis (`visualization.py`, `test_model.py`).
* **Utilities**: Additional helper scripts and example visualizations (`ReLU.py`, `test_raster_opening.py`).

## Requirements

Ensure you have Python 3.x installed with the following libraries:

```bash
numpy
torch
torchvision
rasterio
matplotlib
opencv-python
scikit-learn
tqdm
seaborn
```

Install via:

```bash
pip install -r requirements.txt
```

## Usage

**Training the Model:**

```bash
python train.py --data /path/to/dataset --num_epochs 50 --batch 8 --loss focalloss
```

**Running Inference:**

```bash
python Model_To_Raster.py --data_root /path/to/data --model_path /path/to/model.pth --batch_size 4
```

**Visualization:**

```bash
python visualization.py --data_root /path/to/data --model_path /path/to/model.pth
```

## Licensing

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project includes code adapted from previous work licensed under [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html).

**Supervisors:** Bjørn Wangensteen, Sverre Stikbakke, Eina Bergem Jørgensen

## Contact

* **Authors**: Aslak Brynsrud, Hans Olav Lien Johansen
* **Institution**: NTNU, Department of Manufacturing and Civil Engineering
* **Date of Submission**: May 2024
