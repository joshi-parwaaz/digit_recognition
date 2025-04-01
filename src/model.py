import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import joblib
import struct
import os

# Function to read IDX files (e.g., label files)
def read_idx_labels(file_path):
    with open(file_path, 'rb') as f:
        # Read the magic number and number of items
        magic_number, num_items = struct.unpack(">II", f.read(8))
        # Read the labels into a numpy array
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Load the HOG features
X_train_hog = np.load(r'D:\Work\code\digit_recognition\data\features\X_train_hog.npy')
X_test_hog = np.load(r'D:\Work\code\digit_recognition\data\features\X_test_hog.npy')

# Labels (from earlier processed labels)
y_train = read_idx_labels(r'D:\Work\code\digit_recognition\data\mnist_data\train-labels.idx1-ubyte')  # Read label file
y_test = read_idx_labels(r'D:\Work\code\digit_recognition\data\mnist_data\t10k-labels.idx1-ubyte')  # Read label file

# Train k-NN model
def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn

# Train SVM model
def train_svm(X_train, y_train):
    svm = SVC(kernel="linear")
    svm.fit(X_train, y_train)
    return svm

# Ensure the models directory exists
models_dir = r'D:\Work\code\digit_recognition\models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Train both models
print("Training k-NN model...")
knn_model = train_knn(X_train_hog, y_train)
print("Training SVM model...")
svm_model = train_svm(X_train_hog, y_train)

# Save the trained models to the correct directory
print("Saving k-NN model...")
joblib.dump(knn_model, os.path.join(models_dir, 'knn_model.pkl'))

print("Saving SVM model...")
joblib.dump(svm_model, os.path.join(models_dir, 'svm_model.pkl'))

print("Models trained and saved successfully.")
