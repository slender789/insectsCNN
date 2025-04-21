# Bark Beetle Detection and Segmentation Tool

This project provides a modular pipeline for the detection, segmentation, and classification of bark beetles from images taken inside Lindgren traps. The codebase is designed for flexibility, supporting multiple classification algorithms and robust preprocessing and segmentation steps.

## Project Overview

The main goal of this project is to automate the identification and counting of bark beetles in trap images. The pipeline includes:

- **Preprocessing**: Standardizes and enhances images for further analysis.
- **Segmentation**: Isolates regions of interest (potential beetles) from the background.
- **Classification**: Offers several machine learning and deep learning models to classify segmented regions as beetle or non-beetle.

## Preprocessing

Preprocessing is handled by scripts such as `preprocess.py` and utilities in `utils/image_utils.py`. The typical steps include:

- **Resizing**: Images are resized to a standard size (e.g., 64x64 or 224x224) for model compatibility.
- **Normalization**: Pixel values are scaled to [0, 1] or preprocessed for specific models (e.g., ResNet).
- **Noise Reduction**: Optional noise addition/removal (e.g., Poisson, Gaussian) for data augmentation.
- **Lighting Adjustment**: Scripts like `create_variable_lightning.py` can simulate different lighting conditions.

## Segmentation

Segmentation is performed using the `segmentation.py` script, which extracts candidate regions (potential beetles) from the input images. The process typically involves:

- Thresholding and morphological operations to separate foreground (beetles) from background.
- Filtering by area, aspect ratio, and other geometric properties to reduce false positives.
- Outputting masks or cropped regions for further classification.

## Classification Options

The codebase supports several classification approaches, each implemented in its own script:

- **Artificial Neural Network (ANN)** (`ann.py`): A simple feedforward neural network for binary classification.
- **Support Vector Machine (SVM)** (`kernel_svm.py`): Classical machine learning using SVM with RBF kernel.
- **k-Nearest Neighbors (k-NN)** (`knn.py`): Instance-based learning for beetle detection.
- **Naive Bayes (NV)** (`nv.py`): Probabilistic classification using Gaussian Naive Bayes.
- **Simple CNN** (`cnn_simple.py`): A custom convolutional neural network for image classification.
- **ResNet-based CNN** (`cnn.py`): Deep learning using transfer learning with ResNet50 for high-accuracy classification.

All classifiers use a shared, modular image loading and preprocessing utility (`utils/image_utils.py`) for consistency and code reuse.

## How to Use

1. **Preprocess Images**: Use `preprocess.py` or related scripts to standardize your images.
2. **Segment Images**: Run `segmentation.py` to extract candidate regions.
3. **Classify Regions**: Choose a classifier script (e.g., `cnn.py`, `ann.py`, `kernel_svm.py`) and follow its instructions to train or predict.

Each script is self-contained and can be run directly. Refer to the script's code for specific usage patterns and parameter options.

## Modularity

- Common functions are centralized in `utils/image_utils.py`.
- Each classifier and processing step is in its own script for clarity and flexibility.
- The codebase is structured for easy extension and experimentation with new models or preprocessing techniques.

---

This project provides a complete, modular pipeline for bark beetle detection and segmentation from Lindgren trap images, supporting both classical and deep learning approaches.
