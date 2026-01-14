# 🛡️ Lightweight Deepfake Detection

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch-MobileNetV3-orange.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Project Overview
The objective of this project is to develop an end-to-end pipeline for detecting Deepfakes using a **lightweight architecture (MobileNetV3-Small)**. 

Unlike heavy state-of-the-art models (e.g., Xception), this solution is specifically designed for **resource-constrained environments** (mobile/edge devices), reducing inference time while maintaining high accuracy. The project addresses the growing threat of AI-generated fraud (e.g., the "Fake Brad Pitt" scam) by providing an efficient, deployable detection tool.

## 🏗️ Technical Pipeline

Our approach implements a robust processing chain designed to handle real-world variability:

### 1. Preprocessing (Face Extraction)
We utilize **MTCNN (Multi-task Cascaded Convolutional Networks)** to automatically detect and align faces.
* **Purpose:** Eliminates background noise to focus exclusively on facial artifacts.
* **Output:** Faces are cropped and normalized to **224x224** pixels.

### 2. Training Strategy (Anti-Overfitting)
To counter the small size of the Celeb-DF dataset, a dynamic **Data Augmentation** module is applied during training:
* **Techniques:** Random Rotations ($\pm 15^\circ$), Horizontal Flips, Color Jitter.
* **Result:** Forces the model to learn invariant features rather than memorizing pixel noise.

### 3. Model Architecture
* **Backbone:** `mobilenet_v3_small` (Pre-trained on ImageNet).
* **Edge AI Focus:** Uses *Depthwise Separable Convolutions* to reduce parameter count to $\approx 2.5M$ (vs 23M for ResNet50), making it ideal for mobile CPU inference.
* **Loss Function:** `BCEWithLogitsLoss` with `pos_weight` to handle the 1:10 class imbalance (Real vs. Fake).

---

## 🚀 Demo Application (How to Run)

This application is containerized and includes a **smart download feature**: the model weights (~10MB) are automatically fetched from GitHub Releases upon the first launch.

### Option A: Running Locally (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yennaiv45/12502971_Lightweight-deepfake-detection.git
   cd 12502971_Lightweight-deepfake-detection
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run**
    ```bash
    streamlit run .\src\app_streamlit.py
    ```


