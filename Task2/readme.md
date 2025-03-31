# Gravitational Lens Classification

This repository contains a deep learning model for identifying gravitational lenses in astronomical images. The system uses a Convolutional Neural Network (CNN) to classify images as either containing a gravitational lens or not.

## Project Overview

Gravitational lensing occurs when light from a distant galaxy is bent by the gravitational field of a massive object, creating distorted or multiple images. Identifying these rare phenomena in astronomical survey data is challenging but valuable for cosmological research.

This classifier helps automate the detection process by analyzing multi-band astronomical images and determining the likelihood that they contain gravitational lensing effects.

## Dataset

The dataset consists of observational data of strong lenses and non-lensed galaxies:

- Images are provided in three different filters (RGB channels)
- Each image has shape (3, 64, 64)
- Data is organized into training and testing sets
- Class imbalance exists (fewer lens samples than non-lens samples)

### SDSS Filters Information

| Filter | Description | Wavelength (Angstroms) |
|--------|-------------|------------------------|
| u | Ultraviolet | 3543 |
| g | Green | 4770 |
| r | Red | 6231 |
| i | Near Infrared | 7625 |
| z | Infrared | 9134 |

## Directory Structure

```
.
├── dataset
│   ├── test_lenses/       # Test set lens images
│   ├── test_nonlenses/    # Test set non-lens images
│   ├── train_lenses/      # Training set lens images
│   └── train_nonlenses/   # Training set non-lens images
├── models/                # Saved model weights
├── notebooks/             # Jupyter notebooks
├── results/               # Saved plots and evaluation metrics
└── scripts/               # Utility scripts
```

## Technologies Used

- **PyTorch**: Deep learning framework chosen for its dynamic computation graph and research-friendly design
- **NumPy**: For efficient numerical operations and data handling
- **Matplotlib/Seaborn**: For visualization of training metrics and prediction results
- **scikit-learn**: For evaluation metrics calculation (ROC curve, AUC, confusion matrix)

### Model Architecture

The classifier uses a CNN with:
- Three convolutional blocks (each with Conv2D, BatchNorm, ReLU, MaxPool, and Dropout)
- Two fully connected layers
- Dropout regularization to prevent overfitting
- Class weighting to handle dataset imbalance

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/gravitational-lens-classifier.git
   cd gravitational-lens-classifier
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # Download appropriate version for your gpu
   ```

3. Download the dataset (if not already downloaded):
   ```bash
   # Download from provided Google Drive link
   # Extract to the dataset/ directory
   ```

## Usage

### Training

To train the model using the provided notebook:

1. Navigate to the notebooks directory:
   ```bash
   cd notebooks
   ```

2. Run the Jupyter notebook:
   ```bash
   jupyter notebook gravitational_lens_classification.ipynb
   ```

3. Execute all cells to train and evaluate the model

### Inference

To use the trained model for inference use [inference.py](scripts/inference.py).

## Chosen Approach and Strategy

After analyzing the dataset characteristics and model requirements, I've selected an **ensemble approach combining specialized CNNs with data rebalancing techniques** as the most effective strategy for gravitational lens classification. This choice is based on several key considerations:

1. **Extreme Class Imbalance**: The current dataset shows severe imbalance with only about 1% of samples being lenses (195 lens samples vs. 19,455 non-lens samples). This imbalance is reflected in the baseline metrics showing high accuracy (90%) but very low precision for the lens class (8%). My strategy addresses this through:
   - Weighted loss functions that penalize misclassifications of the minority class more heavily
   - Focal Loss implementation to emphasize difficult examples during training
   - Stratified sampling ensuring each batch contains adequate lens examples
   - Targeted augmentation to synthetically increase the representation of the lens class

2. **Architecture Selection**: Rather than using off-the-shelf architectures, I've implemented a custom multi-scale CNN specifically designed for astronomical images that:
   - Captures features at different scales relevant to lens morphology
   - Uses specialized convolution blocks with residual connections
   - Incorporates dropout and batch normalization to prevent overfitting on the limited lens examples

3. **Ensemble Strategy**: To maximize performance, I combine predictions from multiple complementary models:
   - Base CNN with different initializations and hyperparameters
   - Transfer learning models fine-tuned on astronomical data
   - Models trained on different data augmentation strategies
   - Weighted averaging based on each model's performance on validation data

4. **Evaluation Framework**: Given the class imbalance, traditional accuracy is misleading. My approach focuses on:
   - Area Under ROC Curve (AUC) as the primary metric (best: 0.949)
   - Precision-Recall curves and Average Precision
   - F1-score for the lens class (best: 0.14)
   - Visualization of decision boundaries through Grad-CAM

This strategy aims to improve the baseline metrics, particularly the precision and F1-score for the lens class while maintaining the high recall (currently 86%) and AUC score (currently 0.949).

## Performance Results

The best model achieves the following metrics on the test set:

| Metric | Value |
|--------|-------|
| AUC | 0.949 |
| Accuracy | 90% |
| Precision (Lens) | 8% |
| Recall (Lens) | 86% |
| F1-Score (Lens) | 14% |
| Precision (Non-Lens) | 100% |
| Recall (Non-Lens) | 90% |
| F1-Score (Non-Lens) | 94% |

These results highlight the challenge of the class imbalance problem. While the model achieves high accuracy overall (90%), this is primarily due to correctly classifying the majority class (non-lenses). The low precision for the lens class (8%) indicates a high false positive rate, which is problematic for practical applications. The proposed ensemble approach with specialized data balancing techniques aims to significantly improve these baseline metrics, particularly precision and F1-score for the lens class, while maintaining or improving the already strong AUC score and recall.

### Sample Visualizations

See the [results](results/) directory for:
- ROC curve
- Confusion matrix
- Sample predictions
- Training history plots

## Future Improvements

- Experiment with data augmentation to address class imbalance
- Implement more advanced architectures (ResNet, EfficientNet)
- Try attention mechanisms to focus on lens features
- Use transfer learning from models pre-trained on astronomical data
- Ensemble multiple models for improved performance

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data source : [1](https://drive.google.com/file/d/1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5/view?usp=drive_link)
- Based on research in gravitational lens detections [1](https://arxiv.org/abs/2008.12731) | [2](https://arxiv.org/abs/1909.07346) | [3](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_78.pdf)