import numpy as np
import pickle

import os


def load_and_prepare_data(root_path="cifar-10-batches-py", as_grayscale=False):
    """Load raw data using pickle."""
    # Check if the cifar10 dataset has been downloaded
    train_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_batch = 'test_batch'

    # Load the training data
    try:
        for batch in train_batches:
            batch = os.path.join(root_path, batch)
            with open(batch, 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
                data = batch[b'data']
                labels = batch[b'labels']
                if 'train_data' in locals():
                    train_data = np.concatenate((train_data, data))
                    train_labels = np.concatenate((train_labels, labels))
                else:
                    train_data = data
                    train_labels = labels
    except FileNotFoundError:
        print("The CIFAR-10 dataset has not been downloaded. Download and extract the dataset from https://www.cs.toronto.edu/~kriz/cifar.html and place the data_batch files in the cifar10 directory.")
        return
    
    # Load the test data
    try:
        test_batch = os.path.join(root_path, test_batch)
        with open(test_batch, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
            test_data = batch[b'data']
            test_labels = batch[b'labels']
    except FileNotFoundError:
        print("The CIFAR-10 dataset has not been downloaded. Download and extract the dataset from https://www.cs.toronto.edu/~kriz/cifar.html and place the data_batch files in the cifar10 directory.")
        return
    
    # Reshape the data
    train_data = train_data.reshape((len(train_data), 3, 32, 32)).transpose(0, 2, 3, 1)  # Reshape to (50000, 32, 32, 3)
    test_data = test_data.reshape((len(test_data), 3, 32, 32)).transpose(0, 2, 3, 1)    # Reshape to (10000, 32, 32, 3)

    if as_grayscale:
        # Convert to grayscale using the dot product with the weights [0.299, 0.587, 0.114]
        train_data = np.dot(train_data[..., :3], [0.299, 0.587, 0.114])  # Convert RGB to grayscale
        test_data = np.dot(test_data[..., :3], [0.299, 0.587, 0.114])    # Convert RGB to grayscale

        # Reshape back to (num_samples, height, width, 1)
        train_data = train_data.reshape(-1, 32, 32, 1)  # (50000, 32, 32, 1)
        test_data = test_data.reshape(-1, 32, 32, 1)    # (10000, 32, 32, 1)

    # Reshape the data back to 2D (n_samples, n_features)
    train_data = train_data.reshape(len(train_data), -1)  # Flatten to (50000, n_features)
    test_data = test_data.reshape(len(test_data), -1)     # Flatten to (10000, n_features)

    return train_data, train_labels, test_data, test_labels
