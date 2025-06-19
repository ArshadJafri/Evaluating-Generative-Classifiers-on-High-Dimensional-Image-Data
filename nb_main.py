import numpy as np
from sklearn.metrics import accuracy_score
from models import GaussianNBModel
from utils import load_and_prepare_data  

# Load the CIFAR-10 dataset in both RGB and Grayscale formats
train_data_rgb, train_labels, test_data_rgb, test_labels = load_and_prepare_data(as_grayscale=False)
train_data_gray, _, test_data_gray, _ = load_and_prepare_data(as_grayscale=True)

# Reshape the grayscale data into 2D feature vectors (n_samples, n_features)
train_data_gray = train_data_gray.reshape(len(train_data_gray), -1)
test_data_gray = test_data_gray.reshape(len(test_data_gray), -1)

# Reshape the RGB data into 2D feature vectors (n_samples, n_features)
train_data_rgb = train_data_rgb.reshape(len(train_data_rgb), -1)
test_data_rgb = test_data_rgb.reshape(len(test_data_rgb), -1)

# Initialize and train the Gaussian Naive Bayes model on RGB data
gnb_rgb = GaussianNBModel()
gnb_rgb.fit(train_data_rgb, train_labels)

# Predict on the RGB test data
y_pred_rgb = gnb_rgb.predict(test_data_rgb)

# Calculate and print the accuracy for the RGB dataset
accuracy_rgb = accuracy_score(test_labels, y_pred_rgb)
print(f"RGB Dataset Accuracy: {accuracy_rgb:.2f}")

gnb_gray = GaussianNBModel()
gnb_gray.fit(train_data_gray, train_labels)

y_pred_gray = gnb_gray.predict(test_data_gray)

accuracy_gray = accuracy_score(test_labels, y_pred_gray)
print(f"Grayscale Dataset Accuracy: {accuracy_gray:.2f}")
