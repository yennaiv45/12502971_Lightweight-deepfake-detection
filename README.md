# 12502971 : Lightweight deepfake detection

## Project Overview
The objective is to develop an end-to-end pipeline for detecting Deepfakes using a lightweight architecture (MobileNetV3-Small), specifically designed for resource-constrained environments (mobile/edge devices) and also because it reduces the execution time. 

The approach focuses on optimizing the data pipeline and leveraging acceleration techniques to establish a functional baseline rapidly.

## Metrics & Results

As required by the assignment guidelines, here are the performance metrics obtained on the validation subset:

* **Error Metric:** AUC-ROC (Area Under the Receiver Operating Characteristic Curve). 
* Accuracy was not selected as the primary metric due to significant class imbalance in the dataset. With a ratio of approximately 10 Fakes to 1 Real in Celeb-DF, a naive model predicting solely the majority class would artificially achieve high accuracy (>90%) without learning anything.
    
* **Target Goal:** AUC > 0.85 (Initial baseline target).
* **Achieved Result:** **0.9763** (Validation).

###  Analysis of Results (Robustness Check)
The achieved AUC of **0.9763** is highly significant and scientifically valid, unlike naive implementations that might suffer from data leakage.
* **Strict Evaluation:** I implemented a **Video-Level Split (80/20)** strategy. This ensures that the frames in the Validation set come from videos *never seen* by the model during Training.
* **Generalization:** The slight gap between Training AUC (0.99) and Validation AUC (0.97) confirms that the model is generalizing well and not just memorizing the small dataset, thanks to the heavy **Data Augmentation** pipeline.


***Generalization:** Heavy Data Augmentation (Rotation, Color Jitter, Gaussian Blur) was applied during training to prevent the model from memorizing pixel artifacts, forcing it to learn meaningful deepfake features.


## Work Breakdown (Time Spent)

By optimizing scripts (MTCNN Batch Processing on GPU) and implementing smart data sampling, the development and training time were drastically reduced compared to initial estimates[cite: 1233].

| Task | Estimated | Actual | Notes & Optimizations |
| :--- | :--- | :--- | :--- |
| **Dataset Preparation** | 12h | **3h** | Optimized `preprocess.py`: MTCNN with Batch Processing + Frame skipping (1 frame/sec). |
| **Model Design** | 10h | **3h** | Efficient use of `torchvision` MobileNetV3 (Transfer Learning). |
| **Training** | 15h | **1h** | Training time was reduced to ~20 mins by using the optimized subset and **Mixed Precision** (`torch.amp`) acceleration. |
| **Optimization** | 10h | **1h** | Implementation of `pos_weight` (Class Weights) and Learning Rate adjustment. |
| **Testing** | 8h | **1h30** | Unit testing (`test_model.py`) and technical validation. |
| **Total** | **60h** | **10h** | *Highly efficient "Hacking" pipeline established.* |

## Technical Architecture

## Technical Architecture

### 1. Preprocessing (Data Pipeline)
* **Face Detection & Cropping (MTCNN):** Utilized **MTCNN** (Multi-task Cascaded Convolutional Networks) to automatically detect, align, and crop faces from raw video frames. 
    * *Purpose:* This critical step eliminates background noise (clutter), forcing the model to focus exclusively on facial artifacts where deepfake manipulation occurs.
    * *Output:* Cropped faces are resized to $224 \times 224$ pixels.
* **Smart Hacking Strategy:** To meet the assignment's time constraints, I developed a script that selects a representative subset of videos (approx. 200 videos total) instead of processing the full 500GB dataset.
* **Split Strategy:** Implemented a strict **80/20 Video-Level Split**.
    * *Train:* 80% of unique videos.
    * *Val:* 20% of unique videos.
    * *Benefit:* Prevents Data Leakage (ensures the model never sees frames from validation videos during training).
* **Sampling:** Extracted 10-15 frames per video, uniformly distributed over time, to capture temporal variations without redundancy.

### 2. Model (MobileNetV3-Small)
* **Backbone:** `mobilenet_v3_small` pre-trained on ImageNet.
* **Head:** Replacement of the final classifier with a linear layer for binary classification (Real vs. Fake).
* **Loss Function:** `BCEWithLogitsLoss` for numerical stability, utilizing `pos_weight` to handle class imbalance.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yennaiv45/12502971_Lightweight-deepfake-detection.git
   cd ./12502971_Lightweight-deepfake-detection

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run**
    ```bash
    streamlit run .\src\app_streamlit.py
    ```


