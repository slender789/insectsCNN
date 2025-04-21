import numpy as np
from utils.image_utils import load_images_from_folder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load and preprocess the images
insects_folder = 'all_insects'
no_insects_folder = 'all_noise'

insect_images, insect_labels = load_images_from_folder(insects_folder, label=1)
no_insect_images, no_insect_labels = load_images_from_folder(no_insects_folder, label=0)

# Combine and split the dataset
X = np.array(insect_images + no_insect_images)
y = np.array(insect_labels + no_insect_labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model with RBF kernel
model = SVC(kernel='rbf', C=1.0, gamma=0.01, class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Non-Insect", "Insect"]))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion_matrix:", cm)
