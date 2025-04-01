# Handwritten Digit Recognition

This project aims to recognize handwritten digits using machine learning algorithms. The model is trained on the MNIST dataset, and the primary features are extracted using **Histogram of Oriented Gradients (HOG)**. The project uses two classification algorithms: **k-Nearest Neighbors (k-NN)** and **Support Vector Machines (SVM)**.

---

## Project Overview

The project is organized into several key components:
- **Data Preprocessing**: The MNIST dataset is loaded, and HOG features are extracted from the images.
- **Model Training**: The extracted features are used to train two models: k-Nearest Neighbors (k-NN) and Support Vector Machines (SVM).
- **Evaluation**: The models are evaluated using accuracy metrics and confusion matrices.
- **Results**: The evaluation results, including confusion matrices and accuracy reports, are saved for future reference.

---

## Directory Structure

```
digit_recognition/ 
│
├── data/                # Raw and processed data
│   ├── mnist_data # Raw MNIST dataset (if you download it manually)
│   └── features     # Processed features (e.g., HOG features) 
│
├── src/                 # Code files
│   ├── load_data.py     # Load the dataset
│   ├── feature_extraction.py  # Extract HOG features
│   ├── model.py         # Train models (k-NN, SVM)
│   └── evaluate.py      # Evaluate models (e.g., accuracy, confusion matrix)
│
├── models/              # Saved models (optional)
│   ├── knn_model.pkl    # Saved k-NN model
│   └── svm_model.pkl    # Saved SVM model
│
├── results/             # Evaluation results
│   ├── confusion_matrix.png  # Confusion matrix plot
│   └── accuracy_report.txt  # Text file with model evaluation metrics
│
├── requirements.txt     # List of required packages
├── README.md            # Project overview and instructions
└── .gitignore           # Git ignore file
```

---

## Requirements

The following Python libraries are required to run the project:
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`
- `scikit-image` (for HOG feature extraction)

You can install the required dependencies using:

```
pip install -r requirements.txt
```

---

## Dataset

The project uses the MNIST dataset, which consists of 28x28 pixel grayscale images of handwritten digits (0-9). The dataset is publicly available and can be downloaded [here](http://yann.lecun.com/exdb/mnist/) or you can directly use the provided files in the `data/raw_data/` directory if they are already downloaded.

---

## How to Run the Project

### 1. Data Loading and Preprocessing

To load the MNIST dataset and preprocess the images by extracting HOG features, run the following script:

```
python src/feature_extraction.py
```

This will:
- Load the MNIST dataset.
- Extract HOG features from the images.
- Save the processed features as `.npy` files in the `data/features/` directory.

---

### 2. Train Models

To train the k-NN and SVM models on the extracted features, run the following script:

```
python src/model.py
```

This will:
- Load the HOG features and labels.
- Train the k-NN and SVM models.
- Save the trained models as `knn_model.pkl` and `svm_model.pkl` in the `models/` directory.

---

### 3. Evaluate Models

To evaluate the performance of the models, including generating confusion matrices and calculating accuracy, run:

```
python src/evaluate.py
```

This will:
- Load the trained models.
- Make predictions on the test set.
- Generate and save confusion matrices (`.png`) and an accuracy report (`accuracy_report.txt`) in the `results/` folder.

---

### 4. (Optional) Hyperparameter Tuning

You can use the script `tune_hyperparameters.py` (if available) to tune the hyperparameters of the k-NN and SVM models.

---

### 5. Results and Metrics

The evaluation results (confusion matrices and accuracy reports) will be saved in the `results/` directory. 

---

## Notes

- The models are trained on the MNIST dataset, and HOG features are extracted for better performance.
- The project allows easy experimentation with different models and hyperparameters.
- Results are saved in the `results/` folder for analysis.

---

## License

This project is open-source and available under the [MIT License](LICENSE).
```
