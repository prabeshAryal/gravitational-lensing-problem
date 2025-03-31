# Task 1: Multi-Class Gravitational Lens Substructure Classification

**DeepLense Evaluation Test - Common Test I: Multi-Class Classification**

## Project Overview

This repository contains the implementation and results for Task 1 of the DeepLense Evaluation Test: **Multi-Class Gravitational Lens Substructure Classification**.  The goal is to classify astronomical images into three categories based on the type of gravitational lens substructure (or lack thereof). This task aims to assess the feasibility of distinguishing subtle substructure variations in lens images using deep learning.

## Dataset

The dataset consists of strong lensing images categorized into three classes:

*   **No Substructure ('no'):** Lenses without visible substructure.
*   **Subhalo Substructure ('sphere'):** Lenses with subhalo substructure.
*   **Vortex Substructure ('vort'):** Lenses with vortex substructure.

Dataset details:

*   **Source:** [dataset.zip - Google Drive](https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view)
*   **Format:** 3-channel images (RGB) representing different filters.
*   **Image Shape:** (3, 64, 64)
*   **Normalization:** Images are pre-normalized using min-max normalization.
*   **Structure:** Organized into `dataset/train` and `dataset/val` folders, with class subfolders (`no`, `sphere`, `vort`) within each.
*   **Balanced Dataset:**
    *   **Training Set:** 10,000 samples per class (30,000 total)
    *   **Validation Set:** 2,500 samples per class (7,500 total)

### SDSS Filters Information (Assumed RGB Channels)

| Channel | Filter | Description |
|---------|--------|-------------|
| R       | r      | Red         |
| G       | g      | Green       |
| B       | u      | Ultraviolet |

*Note: While the dataset description mentions SDSS filters, the exact mapping to RGB channels is an assumption for visualization purposes in this context.*

## Directory Structure (Task 1)

```
Task1/
├── dataset/
│   ├── train/
│   │   ├── no/
│   │   ├── sphere/
│   │   └── vort/
│   └── val/
│       ├── no/
│       ├── sphere/
│       └── vort/
├── models/                # Saved model weights
├── notebooks/             # Jupyter notebooks
└── results/               # Plots, evaluation metrics
```

## Technologies Used

-   **PyTorch:** Chosen for its flexibility and strong research community, ideal for CNN development.
-   **NumPy:**  Essential for numerical computations and dataset handling.
-   **Matplotlib/Seaborn:** For creating visualizations of training history and model performance.
-   **scikit-learn:** Used for calculating evaluation metrics like ROC curves, AUC, and classification reports.

### Model Architecture

The classification model is based on a Convolutional Neural Network (CNN) architecture:

*   **Four Convolutional Blocks:**
    *   Each block: `Conv2d`, `BatchNorm2d`, `ReLU`, `MaxPool2d`, `Dropout2d (0.5)`
    *   Filter counts progressively increase: 32, 64, 128, 256.
*   **Two Fully Connected Layers:**
    *   `Linear (512 neurons) + ReLU + Dropout (0.5)`
    *   `Linear (3 neurons)` - Output layer for 3 classes.

## Chosen Approach and Strategy

For Task 1, a standard Convolutional Neural Network (CNN) architecture was chosen as the initial approach for multi-class image classification.  The rationale behind this choice was:

1.  **CNNs are effective for image classification:** CNNs excel at automatically learning hierarchical features from images, making them a suitable starting point for this task.
2.  **Relatively balanced dataset:**  The dataset is balanced across the three classes, reducing the initial need for complex class imbalance handling techniques. This allows for focusing on model architecture and training effectiveness first.
3.  **Simplicity for baseline:**  A straightforward CNN provides a good baseline to evaluate the inherent difficulty of the multi-class substructure classification task. If a standard CNN struggles, it signals the need for more advanced techniques or architectural modifications.

The chosen CNN architecture incorporates:

*   **Multiple Convolutional Layers:** To progressively extract complex features from the 64x64 input images.
*   **Batch Normalization:** To stabilize training and potentially improve generalization.
*   **ReLU Activation:** To introduce non-linearity.
*   **Max Pooling:** To reduce dimensionality and focus on salient features.
*   **Dropout:** As a regularization technique to prevent overfitting.

**Why this approach?**  Given the balanced nature of the dataset and the task description, a standard CNN architecture represented a pragmatic first step. It allowed for a quick implementation and evaluation to understand the baseline performance achievable for differentiating these three classes of substructure.  If performance was satisfactory, further refinements could be explored. If not (as it turned out), it would clearly indicate the need for more sophisticated strategies.

## Performance Results

The model's performance in Task 1 was **poor**, indicating the difficulty of this multi-class classification task with the current approach.

| Metric               | Value    |
|-----------------------|----------|
| **Weighted AUC Score** | **0.5086** |
| Accuracy              | 34%      |
| **Classification Report:** |          |
| Class               | Precision | Recall | F1-score |
| No Substructure     | 0.37      | 0.05   | 0.08     |
| Subhalo             | 0.34      | 0.33   | 0.33     |
| Vortex              | 0.34      | 0.64   | 0.44     |

*   **AUC Score:**  Near 0.5, indicating performance close to random chance.
*   **Accuracy:** Around 34%, also near chance level for a 3-class problem (33.33%).
*   **Classification Report:**  Shows low precision and recall across all classes, with F1-scores also being low.

**Sample Visualizations:**

See the [results folder](results) for visualizations. A [README.md](results/readme.md) file within the `results/` folder provides a quick summary of the generated plots:

*   **`training_history_multi.png`:** Training/Validation Loss and Accuracy curves.
*   **`roc_curve_multi.png`:** Multi-class ROC Curve.
*   **`confusion_matrix_multi.png`:** Confusion Matrix.
*   **`sample_predictions_multi.png`:** Sample Predictions with true and predicted labels.
*   **`filters_layer0.png`:** Visualization of first layer filters.

## Future Improvements

To improve performance in Task 1, several avenues could be explored:

*   **More Complex Architectures:** Experiment with deeper or wider CNNs, or architectures specifically designed for fine-grained image classification (e.g., ResNet, EfficientNet).
*   **Data Augmentation:** Implement data augmentation techniques to increase data variability and potentially improve generalization.
*   **Hyperparameter Tuning:**  Systematically tune hyperparameters (learning rate, weight decay, dropout rates, etc.) to optimize model training.
*   **Advanced Training Techniques:** Explore techniques like transfer learning or fine-tuning pre-trained models if relevant pre-trained weights are available.
*   **Feature Analysis:**  Investigate methods to better understand and potentially enhance the features the model is learning, perhaps through visualization techniques or feature engineering.

## Acknowledgements

-   **Dataset Source:** [dataset.zip - Google Drive](https://drive.google.com/file/d/1ZEyNMEO43u3qhJAwJeBZxFBEYc_pVYZQ/view)
-   **Based on research in gravitational lens detections** (See root `README.md` at [main](../readme.md) for full citations).
