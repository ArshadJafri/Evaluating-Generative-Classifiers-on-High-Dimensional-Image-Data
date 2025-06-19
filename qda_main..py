

import numpy as np
from models import QDAModel as QDA
from utils import load_and_prepare_data
from sklearn.metrics import accuracy_score

def main():
    print("Loading RGB data")
    train_data_rgb, train_labels_rgb, test_data_rgb, test_labels_rgb = load_and_prepare_data(root_path='cifar-10-batches-py', as_grayscale=False)

    qda_rgb = QDA()
    qda_rgb.fit(train_data_rgb.reshape(-1, 3072), train_labels_rgb)  # Reshape to (N_samples, N_features)

    predictions_rgb = qda_rgb.predict(test_data_rgb.reshape(-1, 3072))
    accuracy_rgb = accuracy_score(test_labels_rgb, predictions_rgb)
    print(f'Accuracy on RGB data: {accuracy_rgb:.2f}')

    print("Loading Grayscale data")
    train_data_gray, train_labels_gray, test_data_gray, test_labels_gray = load_and_prepare_data(root_path='cifar-10-batches-py', as_grayscale=True)

    qda_gray = QDA()
    qda_gray.fit(train_data_gray.reshape(-1, 1024), train_labels_gray)  # Reshape to (N_samples, N_features)

    predictions_gray = qda_gray.predict(test_data_gray.reshape(-1, 1024))
    accuracy_gray = accuracy_score(test_labels_gray, predictions_gray)
    print(f'Accuracy on Grayscale data: {accuracy_gray:.2f}')

if __name__ == "__main__":
    main()
