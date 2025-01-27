# Handwritten Digit Recognition with Dimensionality Reduction, Model Calibration, Performance Profiling, and Time Series Cross-Validation

## Project Overview

### Main Objectives

* Develop a comprehensive machine learning pipeline using various algorithms for handwritten digit recognition.
* Enhance the pipeline's performance using dimensionality reduction for feature selection and model calibration for improved accuracy.
* Evaluate model performance comprehensively using detailed performance metrics and benchmarking.
* Implement time series cross-validation to assess model stability over time.

### Key Features

* End-to-end machine learning pipeline with data loading, preprocessing, feature selection, model training, model calibration, and model evaluation.
* Integration of dimensionality reduction techniques to select the most informative features for classification.
* Implementation of model calibration to correct prediction probabilities and improve model reliability.
* Extensive performance evaluation using multiple metrics, including accuracy, precision, recall, F1 score, ROC AUC, and log loss.
* Benchmarking against multiple machine learning algorithms to identify the best performing model for the task.

### Technical Highlights

* Employ dimensionality reduction techniques such as Principal Component Analysis (PCA) or SelectKBest for feature selection.
* Utilize model calibration methods like Platt Scaling or Isotonic Regression to enhance prediction probabilities.
* Implement time series cross-validation to validate model performance over varying time periods.
* Profile model performance in terms of runtime, memory usage, and training time for optimization.

## Technical Details

### Model Architecture

The pipeline employs a variety of machine learning algorithms, including:

* K-Nearest Neighbors
* Decision Tree
* Logistic Regression
* Support Vector Machine
* Random Forest
* Gradient Boosting
* AdaBoost
* XGBoost (if available)
* LightGBM (if available)
* CatBoost (if available)

### Data Processing Pipeline

The data preprocessing pipeline consists of the following steps:

* Data loading and splitting into training and testing sets.
* Feature scaling using Standard Scaling or Robust Scaling.
* Dimensionality reduction using PCA or SelectKBest.

### Key Algorithms

**Dimensionality Reduction:**
* Principal Component Analysis (PCA)
* SelectKBest

**Model Calibration:**
* Platt Scaling
* Isotonic Regression

**Time Series Cross-Validation:**
* TimeSeriesSplit from scikit-learn

## Performance Metrics

### Evaluation Results

The pipeline evaluates models using the following metrics:

* Accuracy
* Precision
* Recall
* F1 score
* ROC AUC
* Log Loss

### Benchmark Comparisons

The pipeline benchmarks model performance against multiple algorithms to identify the best performing model for the task.

### Model Strengths

The pipeline's strengths include:

* Comprehensive feature selection and model calibration techniques to enhance accuracy.
* Extensive performance evaluation metrics for detailed model assessment.
* Time series cross-validation for assessing model stability over time.
* Benchmarking against multiple algorithms for optimal model selection.

## Implementation Details

### Dependencies

* scikit-learn (>=0.24)
* pandas (>=1.1)
* numpy (>=1.19)
* matplotlib (>=3.3)
* seaborn (>=0.11)
* openml (>=0.9)
* time series split-validate (>=0.3)

### System Requirements

* Python 3.6 or later
* Operating system: Windows, Linux, or macOS

### Setup Instructions

1. Install the required dependencies.
2. Clone the repository.
3. Run `python main.py` to execute the pipeline.