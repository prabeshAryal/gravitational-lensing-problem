# DeepLense Gravitational Lens Classifier - GSoC 2025 Evaluation Project

**Tests as a Prospective GSoC 2025 Applicant - DeepLense Project**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%20%E2%82%892.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and implementation for the DeepLense Gravitational Lens Classifier project, developed as part of the evaluation tests for prospective Google Summer of Code (GSoC) 2025 applicants interested in the DEEPLENSE project under the ML4SCI organization.

The project addresses the challenge of automatically identifying gravitational lenses in astronomical images using deep learning techniques, specifically focusing on Convolutional Neural Networks (CNNs) implemented in PyTorch.

**This repository includes solutions for two tasks:**

*   **Task I: Multi-Class Classification** - Classifying images into three categories: no substructure, subhalo substructure, and vortex substructure. (See [Task1/readme.md](Task1/readme.md) for details)
*   **Task II: Lens Finding (Binary Classification)** - Identifying gravitational lenses versus non-lensed galaxies. (See [Task2/readme.md](Task2/readme.md) for details)


**GSoC Project Details:**

*   **Project Title:** Physics Guided Machine Learning on Real Lensing Images
*   **Project Length:** 175 / 350 hours
*   **Difficulty Level:** Intermediate/Advanced
*   **Expected Results:** A robust architecture capable of handling diverse lensing images from real galaxy datasets, providing insights into lensing systems and substructures.
*   **Requirements:** Python, PyTorch, Machine Learning experience, Computer Vision knowledge, Familiarity with Autoencoders.
*   **Mentors:** Michael Toomey (MIT), Sergei Gleyzer (University of Alabama)


## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/prabesharyal/gravitational-lens-classifier.git
    cd gravitational-lens-classifier
    ```

2.  **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Linux/macOS
    venv\Scripts\activate  # On Windows

    pip install -r requirements.txt

    # Install PyTorch with CUDA (replace with your CUDA version if needed)
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 

    jupyter lab #then navigate within browser
    ```
    *   **Note:** Adjust the PyTorch installation command (`pip3 install torch torchvision torchaudio ...`) to match your CUDA version and operating system as needed from the [PyTorch website](https://pytorch.org/get-started/locally/). If you don't have a CUDA-enabled GPU, you can install the CPU-only version.

3.  **Navigate to Task Folders:**

    *   For Task I: `cd Task1` (see [Task1/README.md](Task1/README.md) inside Task1 folder)
    *   For Task II: `cd Task2` (see [Task2/README.md](Task2/README.md) inside Task2 folder)

## Repository Structure

```
gravitational-lens-classifier/
├── .gitignore
├── README.md                  # Landing README (this file)
├── requirements.txt         # Python dependencies
├── Task1/                     # Task I: Multi-Class Classification
│   ├── dataset/             # Dataset for Task I
│   ├── models/              # Saved models for Task I
│   ├── notebooks/           # Jupyter Notebooks for Task I
│   │   └── multi-class-classification.ipynb
│   ├── results/             # Results and visualizations for Task I
│   └── README.md            # README for Task I
└── Task2/                     # Task II: Lens Finding (Binary Classification)
    ├── dataset/             # Dataset for Task II
    ├── models/              # Saved models for Task II
    ├── notebooks/           # Jupyter Notebooks for Task II
    │   └── gravitational_lens_classification.ipynb
    ├── results/             # Results and visualizations for Task II
    ├── scripts/             # Optional scripts for Task II
    └── README.md            # README for Task II
```

## Tasks Overview

*   **Task I: Multi-Class Classification ([Task1/README.md](Task1/README.md))**
    *   Goal: Classify gravitational lens images into 'no', 'sphere', and 'vort' substructure classes.
    *   Dataset: Download and place in `Task1/dataset/`
    *   Implementation: [Task1/notebooks/multi-class-classification.ipynb](Task1/notebooks/multi-class-classification.ipynb)
    *   Results: [Task1/results/](Task1/results/)

*   **Task II: Lens Finding (Binary Classification) ([Task2/README.md](Task2/README.md))**
    *   Goal: Identify gravitational lenses vs. non-lensed galaxies (binary classification).
    *   Dataset: Download and place in `Task2/dataset/`
    *   Implementation: [Task2/notebooks/gravitational_lens_classification.ipynb](Task2/notebooks/gravitational_lens_classification.ipynb)
    *   Results: [Task2/results/](Task2/results/)

## Acknowledgements

-   **Gravitational Lens Detection Research Basis:**
    *   [Paper 1](https://arxiv.org/abs/2008.12731)
    *   [Paper 2](https://arxiv.org/abs/1909.07346)
    *   [Paper 3](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_78.pdf)


**Contact:** ml4-sci@cern.ch (for GSoC 2025 application related queries - **DO NOT contact mentors/repo owners directly**)

**Author:** Prabesh Aryal