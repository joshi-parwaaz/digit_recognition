import numpy as np
from skimage.feature import hog
import cv2
import os

# Absolute paths to the directories for saving features
FEATURES_DIR = r'D:\Work\code\digit_recognition\data\features'

# Function to extract HOG features from an image
def extract_hog_features(image):
    # Reshape the image from a 1D vector to a 28x28 image
    image = image.reshape(28, 28).astype(np.uint8)
    
    # Extract HOG features from the image
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

# Function to extract HOG features for the entire dataset
def extract_features(X):
    features = []
    for image in X:
        features.append(extract_hog_features(image))
    return np.array(features)

if __name__ == "__main__":
    # Load data (Ensure this is correctly working)
    from load_data import load_data
    X_train, y_train, X_test, y_test = load_data()

    # Extract HOG features for training and testing data
    X_train_hog = extract_features(X_train)
    X_test_hog = extract_features(X_test)

    # Ensure the features directory exists
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)

    # Save the extracted features to files
    np.save(os.path.join(FEATURES_DIR, 'X_train_hog.npy'), X_train_hog)
    np.save(os.path.join(FEATURES_DIR, 'X_test_hog.npy'), X_test_hog)

    print(f"Extracted HOG features for training data: {X_train_hog.shape}")
    print(f"Extracted HOG features for test data: {X_test_hog.shape}")
