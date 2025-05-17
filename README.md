

# Advanced Deep Learning Project: Urban Scene Segmentation with Fast-SCNN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/) [![PyTorch Version](https://img.shields.io/badge/pytorch-1.9%2B-orange.svg)](https://pytorch.org/) This repository showcases an advanced deep learning project focusing on **Urban Scene Segmentation using Fast-SCNN**: a lightweight and efficient model for real-time semantic segmentation of urban environments.

This project was developed as part of advanced coursework in Neural Networks and Deep Learning, focusing on cutting-edge architectures and their practical implementation for computer vision tasks.

## Table of Contents
1.  [Project: Urban Image Segmentation with Fast-SCNN](#project-urban-image-segmentation-with-fast-scnn)
    * [1.1. Objective](#11-objective)
    * [1.2. Fast-SCNN Model Overview](#12-fast-scnn-model-overview)
    * [1.3. Dataset: CamVid](#13-dataset-camvid)
    * [1.4. Methodology & Implementation Highlights](#14-methodology--implementation-highlights)
    * [1.5. Results & Evaluation](#15-results--evaluation)
2.  [Technology Stack](#technology-stack)
3.  [Acknowledgments](#acknowledgments)

---

## Project: Urban Image Segmentation with Fast-SCNN
*Relevant Keywords: Semantic Segmentation, Urban Scenes, Fast-SCNN, Real-Time Deep Learning, Computer Vision, CamVid Dataset, PyTorch*

### 1.1. Objective
To implement and evaluate the Fast-SCNN (Fast Semantic Segmentation Network) model for real-time, high-resolution semantic segmentation of urban street scenes. The primary dataset used for this task was the Cambridge-driving Labeled Video Database (CamVid).

### 1.2. Fast-SCNN Model Overview
Fast-SCNN is designed for efficiency and speed without a significant compromise in accuracy, making it suitable for applications on resource-constrained devices.
-   **Architecture**: It features a multi-branch architecture with a shared "learning to downsample" module that feeds into two distinct branches:
    -   **Detail Path**: Captures high-resolution, low-level spatial details.
    -   **Context Path**: Processes lower-resolution features to capture global context using components like Inverted Residual Blocks (from MobileNetV2) and a Pyramid Pooling Module (PPM).
-   **Feature Fusion**: A lightweight Feature Fusion Module (FFM) efficiently merges the high-resolution details from the detail path and the contextual information from the context path.
-   **Efficiency**: Achieved through extensive use of Depthwise Separable Convolutions (DSConv) and a shallow architecture, resulting in a model with only approximately **1.11 million parameters**. This allows for significantly faster inference times (e.g., reported 123.5 FPS on 1024x2048 images) compared to traditional encoder-decoder models like U-Net or FCN.
-   **Training**: Can be trained effectively from scratch, reducing dependency on pre-trained backbones.

### 1.3. Dataset: CamVid
The CamVid dataset was utilized for training and evaluating the Fast-SCNN model.
-   **Training Samples**: 367
-   **Validation Samples**: 101
-   **Test Samples**: 233
-   **Preprocessing**: Involved organizing image and mask folders, handling damaged masks, and implementing a `JointTransform` class for consistent data augmentation (e.g., RandomResizedCrop, Rotate, Flip, ColorJitter) on both images and their corresponding segmentation masks.

### 1.4. Methodology & Implementation Highlights
-   **Optimizer**: `AdamW` (learning rate: 1e-3, weight decay: 1e-5).
-   **Learning Rate Scheduler**: `OneCycleLR` (max_lr: 5e-4, pct_start: 0.3).
-   **Loss Function**: Weighted `CrossEntropyLoss` (weights calculated based on inverse class frequency). Experiments were also conducted with label smoothing and Focal Loss.
-   **Metrics**: Pixel Accuracy, mean Intersection over Union (mIoU), and Dice Coefficient.
-   **Key Implemented Components**:
    -   Depthwise Separable Convolution (DSConv)
    -   Inverted Residual Blocks
    -   Pyramid Pooling Module (PPM)
-   **Training Challenges**: Addressed challenges such as hyperparameter tuning for `OneCycleLR` and managing oscillations in validation loss. Gradient clipping (`max_norm=1.0`) was used for stability. Early stopping (patience: 10 epochs based on validation mIoU) was employed, with training stopping at epoch 94.

### 1.5. Results & Evaluation
The model's performance was evaluated on the CamVid test set:
-   **Mean Intersection over Union (mIoU)**: `0.3975`
-   **Dice Coefficient**: `0.4952`
-   **Pixel Accuracy**: `0.7779`

**Analysis**: The implemented Fast-SCNN model demonstrated reasonable performance for a lightweight, real-time architecture. Dominant classes (e.g., Sky, Road, Building) were generally well-detected. Challenges were observed with less frequent classes and complex boundaries, particularly in shaded areas. The drop in mIoU from validation (~0.55) to test (~0.40) suggests areas for improvement in model generalization.

---

## Technology Stack
-   **Programming Language**: Python (3.8+)
-   **Deep Learning Framework**: PyTorch (1.9+)
-   **Core Libraries**: NumPy, OpenCV, Matplotlib, Pillow
-   **Dataset Handling**: Pandas (optional, for metadata)
-   **Development Environment**: Jupyter Notebooks, VS Code (or your preferred IDE)
-   **(Mention any other significant tools or libraries used, e.g., specific data augmentation libraries, CUDA for GPU acceleration)**

---




## Acknowledgments
-   This work was performed as part of the "Neural Networks and Deep Learning" course at the Faculty of Electrical and Computer Engineering, University of Tehran.
-   We thank the instructors and teaching assistants for their valuable guidance and support.
-   Gratitude to the creators of the CamVid dataset.
-   Inspiration and foundational concepts were drawn from the original research paper for [Fast-SCNN](https://arxiv.org/pdf/1902.04502).

---

*To further enhance SEO and discoverability for your repository, consider adding relevant "topics" on GitHub. Go to your repository page and click the "Manage topics" button. Suggested topics: `deep-learning`, `computer-vision`, `image-segmentation`, `fast-scnn`, `pytorch`, `camvid-dataset`, `academic-project`.*
