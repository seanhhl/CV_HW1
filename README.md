# NYCU Computer Vision 2026 HW1

## Introduction
This repository contains the **PyTorch** implementation for **NYCU Computer Vision 2026 HW1: Image Classification**. The goal is to classify RGB images into 100 object categories under a strict **<100M parameter limit**. 

We utilize an **SE-ResNeXt-101** backbone combined with advanced training techniques to effectively prevent overfitting and achieve high accuracy:
* **Label Smoothing**
* **Cosine Annealing LR**
* **Data Augmentation**

---

## Environment Setup
This project is implemented and fully tested on **Google Colab**.

### Basic Requirements:
* **Python:** 3.8+
* **Libraries:** `torch`, `torchvision`, `Pillow`
* **Hardware:** CUDA-enabled GPU (e.g., Colab T4/L4 GPU)

---

## Usage
Please make sure your dataset is placed in the correct directory (e.g., `./dataset_folder/`) before running the scripts.

### 1. Training the Model
To train the SE-ResNeXt-101 model, simply run:
```bash
python train.py
```

The trained weights will be saved as se_resnext101_advanced_weights.pth automatically.

### 2. Running Inference (Prediction)

To generate predictions on the test dataset and output the result as submission:
```bash
python predict.py
```

## Snapshot

<img width="946" height="310" alt="image" src="https://github.com/user-attachments/assets/b952476c-1d64-41dc-9768-5d3a4ffebfa0" />
