import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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

# Load the models
knn_model = joblib.load(r'D:\Work\code\digit_recognition\models\knn_model.pkl')
svm_model = joblib.load(r'D:\Work\code\digit_recognition\models\svm_model.pkl')

# Load test data and labels
X_test_hog = np.load(r'D:\Work\code\digit_recognition\data\features\X_test_hog.npy')
y_test = read_idx_labels(r'D:\Work\code\digit_recognition\data\mnist_data\t10k-labels.idx1-ubyte')  # Read the label file

# Predict using k-NN and SVM
y_pred_knn = knn_model.predict(X_test_hog)
y_pred_svm = svm_model.predict(X_test_hog)

# Calculate accuracy
accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Ensure the results directory exists
results_dir = r'D:\Work\code\digit_recognition\results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Confusion matrix for k-NN
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - k-NN')
plt.savefig(os.path.join(results_dir, 'confusion_matrix_knn.png'))

# Confusion matrix for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - SVM')
plt.savefig(os.path.join(results_dir, 'confusion_matrix_svm.png'))

# Print accuracy results
print(f"Accuracy of k-NN: {accuracy_knn}")
print(f"Accuracy of SVM: {accuracy_svm}")

# Save accuracy report
with open(os.path.join(results_dir, "accuracy_report.txt"), "w") as f:
    f.write(f"Accuracy of k-NN: {accuracy_knn}\n")
    f.write(f"Accuracy of SVM: {accuracy_svm}\n")
