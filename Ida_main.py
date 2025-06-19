import numpy as np
from models import LDAModel as LDA
from utils import load_and_prepare_data

def evaluate_model(train_data, train_labels, test_data, test_labels):

    model = LDA()
    model.fit(train_data, train_labels)
    predictions = model.predict(test_data)
    accuracy = np.mean(predictions == test_labels)
    return accuracy

def main():
    print("Loading RGB data")
    train_data_rgb, train_labels_rgb, test_data_rgb, test_labels_rgb = load_and_prepare_data(root_path='cifar-10-batches-py', as_grayscale=False)
    accuracy_rgb = evaluate_model(train_data_rgb, train_labels_rgb, test_data_rgb, test_labels_rgb)
    print(f"Accuracy on RGB data: {accuracy_rgb:.2f}")
    print("Loading Grayscale data...")
    train_data_gray, train_labels_gray, test_data_gray, test_labels_gray = load_and_prepare_data(root_path='cifar-10-batches-py', as_grayscale=True)
    accuracy_gray = evaluate_model(train_data_gray, train_labels_gray, test_data_gray, test_labels_gray)
    print(f"Accuracy on Grayscale data: {accuracy_gray:.2f}")


if __name__ == "__main__":
    main()
