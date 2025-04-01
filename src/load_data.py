import os
import numpy as np
import struct

# Absolute paths to the MNIST data files
TRAIN_IMAGES_PATH = r'D:\Work\code\digit_recognition\data\mnist_data\train-images.idx3-ubyte'
TRAIN_LABELS_PATH = r'D:\Work\code\digit_recognition\data\mnist_data\train-labels.idx1-ubyte'
TEST_IMAGES_PATH = r'D:\Work\code\digit_recognition\data\mnist_data\t10k-images.idx3-ubyte'
TEST_LABELS_PATH = r'D:\Work\code\digit_recognition\data\mnist_data\t10k-labels.idx1-ubyte'

# Function to load images from the MNIST dataset
def load_images(file_path):
    with open(file_path, 'rb') as f:
        # Skip the magic number and read the metadata
        f.read(4)  # Magic number
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows * num_cols)
        
    return images

# Function to load labels from the MNIST dataset
def load_labels(file_path):
    with open(file_path, 'rb') as f:
        f.read(8)  # Skip the magic number and read the metadata
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels

# Function to load the MNIST dataset
def load_data():
    # Load training and test data
    X_train = load_images(TRAIN_IMAGES_PATH)
    y_train = load_labels(TRAIN_LABELS_PATH)
    X_test = load_images(TEST_IMAGES_PATH)
    y_test = load_labels(TEST_LABELS_PATH)
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
