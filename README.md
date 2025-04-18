# insectsCNN

This project focuses on insect detection using a convolutional neural network (CNN). Below are the main files of the project:

- **Image Files**: Files starting with `all_*` contain the images used to train the AI model.

- **`use_weights.py`**: Uses the trained model weights to predict the category of a given image.

- **`preprocess_raw_image.py`**: Performs preprocessing on the initial image, preparing the data for segmentation.

## Data

1. **Execution Times**: The average segmentation time per image is 0.81 seconds, while the average time to make predictions on each segment per image is 10.29 seconds.

2. **Models**: The models trained so far are CNN, ANN, KNN, NV, and Kernel SVM.

## Accuracy by Model

1. **CNN**: Accuracy: 0.9861.

2. **ANN**: Accuracy: 0.9722.

3. **KNN**: Accuracy: 0.9444.

4. **NV**: Accuracy: 0.9166.

5. **Kernel SVM**: Accuracy: 0.9583.

## CNN Accuracy by Preprocessing Used in Training

| Kernel          | Accuracy  | Loss            |
| --------------- | ----------| --------------- |
 Without          | 96%       | 13.14%          |
 Salt & Pepper    | 100%      | 2.7%            |
 Poisson          | 100%      | 4.43%           |
 All 3            | 98.61%    | 15.91%          |

## Classification Report for CNN Model

|                 | Precision  | Recall    | F1-Score  | Support   |
| --------------- | -----------| ----------| ----------| ----------|
| Not a Beetle    | 1.00       | 0.92      | 0.96      | 12        |
| Beetle          | 0.98       | 1.00      | 0.99      | 60        |
| Accuracy        |            |           | 0.99      | 72        |
| Macro Avg       | 0.99       | 0.96      | 0.97      | 72        |
| Weighted Avg    | 0.99       | 0.99      | 0.99      | 72        |
