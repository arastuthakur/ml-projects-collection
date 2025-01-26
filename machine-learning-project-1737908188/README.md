## Machine Learning Project

### Project Overview

#### Main Objectives

- Demonstrate a comprehensive machine learning pipeline for classification with feature selection, ensemble methods, error analysis, and custom scoring metrics.
- Evaluate multiple models and compare their performance using various metrics.

#### Key Features

- Data preprocessing and feature selection using chi-square test.
- Ensemble methods including Random Forest, AdaBoost, Gradient Boosting, and XGBoost.
- Detailed error analysis using confusion matrix, classification report, and ROC curves.
- Custom scoring metric implementation for specific business requirements.

#### Technical Highlights

- Pipeline architecture for modular and efficient data processing and model training.
- Grid search with cross-validation for optimal model hyperparameter tuning.
- Comprehensive evaluation metrics to quantify model performance and identify areas for improvement.

## Technical Details

### Model Architecture

The project utilizes various classification models, including:

- Linear models: Logistic Regression, Ridge Regression, SGDClassifier
- Ensemble models: Random Forest, AdaBoostClassifier, GradientBoostingClassifier, XGBoostClassifier, LightGBMClassifier, CatBoostClassifier
- Tree-based models: DecisionTreeClassifier
- Support Vector Machines: SVC

### Data Processing Pipeline

The data preprocessing pipeline consists of the following steps:

- **Data Loading and Preprocessing:** Loading the data into a pandas DataFrame and handling missing values and outliers.
- **Feature Selection:** Selecting the most informative features using the chi-square test, implemented using the SelectKBest class.
- **Scaling:** Scaling numerical features within the range [0, 1] using StandardScaler or RobustScaler.

### Key Algorithms

- **Ensemble Methods:** Ensemble methods combine multiple base learners to improve overall performance. Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, and CatBoost are employed.
- **Custom Scoring Metric:** A custom scoring metric based on domain-specific requirements is defined and used for model evaluation.

## Performance Metrics

### Evaluation Results

The models are evaluated using various metrics:

- **Classification Metrics:** Accuracy, precision, recall, F1-score, Kappa, Matthews Correlation Coefficient, and Balanced Accuracy.
- **Error Analysis Metrics:** Confusion matrix, classification report, and receiver operating characteristic (ROC) curve.

### Benchmark Comparisons

The model performances are compared against a baseline model (e.g., Logistic Regression) and other industry benchmarks. This analysis helps identify the most effective models for the given problem.

### Model Strengths

- **Ensemble Methods:** Ensemble methods consistently outperform individual base learners, demonstrating the benefits of combining multiple perspectives.
- **Feature Selection:** Feature selection improves model interpretability and reduces overfitting, leading to better generalization.
- **Custom Scoring Metric:** The custom metric allows for tailored evaluation based on specific business requirements.

## Implementation Details

### Dependencies

- Python 3.8+
- Pandas
- Numpy
- Scikit-learn
- Seaborn
- Matplotlib

### System Requirements

- Windows, macOS, or Linux operating system
- Python 3.8+ installed
- GPU (optional, recommended for faster training)

### Setup Instructions

1. Install the required packages: `pip install -r requirements.txt`.
2. Download the dataset and place it in the `data` directory.
3. Run the main script: `python main.py`.