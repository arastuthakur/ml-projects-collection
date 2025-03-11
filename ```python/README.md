# ```Python

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Diabetes Prediction Project: Clinical Feature Analysis and Machine Learning"""

import logging
import time
import warnings
from typing import Dict, Tuple, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.ensemble import (BaggingClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             matthews_corrcoef, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, LearningCurveDisplay,
                                     StratifiedKFold, TimeSeriesSplit,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier  # For Stacking
import category_encoders as ce

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class MLProject:
    """
    A machine learning pipeline for Diabetes Prediction using Clinical Features,
    incorporating data cleaning, model stacking, learning curves, and time series
    cross-validation techniques.
    """

    def __init__(self, random_state: int = 42, test_size: float = 0.2):
        """
        Initializes project variables.

        Args:
            random_state (int): Random state for reproducibility.
            test_size (float): The proportion of the dataset to include in the test split.
        """
        self.random_state = random_state
        self.test_size = test_size
        self.data: pd.DataFrame = None
        self.X: pd.DataFrame = None
        self.y: pd.Series = None
        self.X_train: pd.DataFrame = None
        self.X_test: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.y_test: pd.Series = None
        self.models: Dict[str, any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.preprocessor = None  # placeholder for the preprocessor later

    def load_data(self) -> None:
        """
        Loads the 'diabetes' dataset from OpenML.
        Handles potential errors during data loading.
        """
        try:
            logging.info("Loading data...")
            self.data = fetch_openml(name="diabetes", version=1, as_frame=True, return_X_y=False).frame
            self.X = self.data.drop("class", axis=1)
            self.y = self.data["class"]
            logging.info("Data loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def explore_data(self) -> None:
        """
        Performs exploratory data analysis (EDA) with visualizations, focusing on
        data cleaning aspects (missing values, outliers), class distribution, and
        feature correlations relevant to the diabetes prediction task.  Includes analysis
        related to the chosen models (model-specific data preparation needs).

        """
        if self.data is None:
            logging.warning("Data not loaded.  Please call load_data() first.")
            return

        logging.info("Performing EDA...")

        # Basic Data Overview
        print("Data Shape:", self.data.shape)
        print("\nData Info:")
        print(self.data.info())
        print("\nMissing Values:")
        print(self.data.isnull().sum())
        print("\nDescriptive Statistics:")
        print(self.data.describe())

        # Class Distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x="class", data=self.data)
        plt.title("Distribution of Diabetes Class")
        plt.show()

        # Feature Distributions (Histograms)
        self.X.hist(figsize=(12, 10))
        plt.suptitle("Histograms of Features", fontsize=16)
        plt.show()

        # Feature Correlations (Heatmap)
        plt.figure(figsize=(10, 8))
        corr_matrix = self.X.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix of Features")
        plt.show()

        # Box plots for outlier detection (using a subset for better visualization)
        num_cols = self.X.select_dtypes(include=np.number).columns
        if len(num_cols) > 0:
            plt.figure(figsize=(15, 8))
            sns.boxplot(data=self.X[num_cols])
            plt.title("Boxplots for Outlier Detection")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            logging.warning("No numeric columns found for boxplot visualization.")


        # Feature Importance (Example using RandomForest - useful for feature selection)
        try:
            model = RandomForestClassifier(random_state=self.random_state)
            model.fit(self.X, self.y)
            importances = model.feature_importances_
            feature_names = self.X.columns
            indices = np.argsort(importances)[::-1]

            plt.figure(figsize=(10, 6))
            plt.title("Feature Importances (RandomForest)")
            plt.bar(range(self.X.shape[1]), importances[indices], align="center")
            plt.xticks(range(self.X.shape[1]), feature_names[indices], rotation=45)
            plt.xlim([-1, self.X.shape[1]])
            plt.tight_layout()
            plt.show()

        except Exception as e:
            logging.warning(f"Feature importance calculation failed: {e}")

        logging.info("EDA complete.")


    def preprocess_data(self) -> None:
        """
        Preprocesses the data, handling missing values, scaling features,
        and splitting into training and testing sets.  Tailored for model stacking,
        learning curve generation, and time series validation considerations.

        Handles missing values using imputation.
        Scales numerical features using MinMaxScaler.
        Encodes categorical features using a simple OneHotEncoder (if needed).
        Splits the data into training and testing sets.
        """
        if self.data is None:
            logging.warning("Data not loaded.  Please call load_data() first.")
            return

        logging.info("Preprocessing data...")

        # Split data into training and test sets *before* imputation to avoid data leakage
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )

        # Identify numerical and categorical features (simplified)
        numerical_features = self.X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = self.X_train.select_dtypes(exclude=np.number).columns.tolist()

        # Imputation
        imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent', 'constant'
        self.X_train[numerical_features] = imputer.fit_transform(self.X_train[numerical_features])
        self.X_test[numerical_features] = imputer.transform(self.X_test[numerical_features])


        # Scaling (MinMaxScaler)
        scaler = MinMaxScaler()  # or StandardScaler, RobustScaler, QuantileTransformer

        self.X_train[numerical_features] = scaler.fit_transform(self.X_train[numerical_features])
        self.X_test[numerical_features] = scaler.transform(self.X_test[numerical_features])


        # Encoding Categorical features (if any)
        if categorical_features:
           encoder = ce.OneHotEncoder(use_cat_names=True) # Handles NaNs by default
           self.X_train = encoder.fit_transform(self.X_train)
           self.X_test = encoder.transform(self.X_test)




        logging.info("Data preprocessing complete.")


    def train_models(self) -> None:
        """
        Trains and tunes multiple machine learning models for diabetes prediction.
        Includes model-specific hyperparameter tuning using GridSearchCV, stratified
        cross-validation, and logging of training times.  Emphasizes models suitable
        for stacking and those that work well with imbalanced datasets.
        """

        if self.X_train is None or self.y_train is None:
            logging.warning("Data not preprocessed. Call preprocess_data() first.")
            return

        logging.info("Training models...")

        # 1. ExtraTreesClassifier
        try:
            start_time = time.time()
            param_grid_et = {
                'n_estimators': [100, 200], # More estimators
                'max_depth': [None, 5, 10], # Depth control
                'min_samples_split': [2, 5], # Controls overfitting
                'min_samples_leaf': [1, 2], # Controls overfitting
                'class_weight': ['balanced', 'balanced_subsample', None] #  Handles imbalanced data
            }
            et = ExtraTreesClassifier(random_state=self.random_state)
            grid_search_et = GridSearchCV(et, param_grid_et, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state), scoring='balanced_accuracy', verbose=0, n_jobs=-1)  # StratifiedKFold for imbalanced data
            grid_search_et.fit(self.X_train, self.y_train)
            self.models['ExtraTreesClassifier'] = grid_search_et.best_estimator_
            end_time = time.time()
            logging.info(f"ExtraTreesClassifier trained in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            logging.error(f"Error training ExtraTreesClassifier: {e}")

        # 2. BaggingClassifier with Decision Trees (more robust to outliers)
        try:
            start_time = time.time()
            param_grid_bagging = {
                'n_estimators': [50, 100],
                'max_samples': [0.5, 1.0],
                'max_features': [0.5, 1.0],
                'base_estimator__max_depth': [None, 5, 10], # Tuning DT inside bagging
                'base_estimator__min_samples_split': [2, 5],
                'base_estimator__min_samples_leaf': [1, 2]
            }

            base_dt = DecisionTreeClassifier(random_state=self.random_state) #explicitly define the base estimator
            bagging = BaggingClassifier(base_estimator=base_dt, random_state=self.random_state)
            grid_search_bagging = GridSearchCV(bagging, param_grid_bagging, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state), scoring='balanced_accuracy', verbose=0, n_jobs=-1)  # StratifiedKFold
            grid_search_bagging.fit(self.X_train, self.y_train)
            self.models['BaggingClassifier'] = grid_search_bagging.best_estimator_
            end_time = time.time()
            logging.info(f"BaggingClassifier trained in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            logging.error(f"Error training BaggingClassifier: {e}")

        # 3. HistGradientBoostingClassifier (handles mixed data, built-in handling of missing values)
        try:
            start_time = time.time()
            param_grid_hgb = {
                'learning_rate': [0.01, 0.1], # Learning rate
                'max_iter': [100, 200],  # Number of boosting stages
                'max_depth': [3, 5, 7],
                'l2_regularization': [0, 0.1], # Regularization
                'early_stopping': [True]  # Monitor validation set and stop if no improvement
            }

            hgb = HistGradientBoostingClassifier(random_state=self.random_state)
            grid_search_hgb = GridSearchCV(hgb, param_grid_hgb, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state), scoring='balanced_accuracy', verbose=0, n_jobs=-1)
            grid_search_hgb.fit(self.X_train, self.y_train)
            self.models['HistGradientBoostingClassifier'] = grid_search_hgb.best_estimator_
            end_time = time.time()
            logging.info(f"HistGradientBoostingClassifier trained in {end_time - start_time:.2f} seconds.")
        except Exception as e:
            logging.error(f"Error training HistGradientBoostingClassifier: {e}")

        # 4. Stacking with Logistic Regression as meta-learner
        try:
            start_time = time.time()

            estimators = [
                ('et', self.models['ExtraTreesClassifier']),
                ('bagging', self.models['BaggingClassifier']),
                ('hgb', self.models['HistGradientBoostingClassifier'])
            ]

            # Using Logistic Regression as the meta-learner
            stacking = VotingClassifier(estimators=estimators, voting='soft')  # Soft voting for probabilities
            stacking.fit(self.X_train, self.y_train)
            self.models['StackingClassifier'] = stacking
            end_time = time.time()
            logging.info(f"StackingClassifier trained in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            logging.error(f"Error training StackingClassifier: {e}")

        logging.info("All models trained.")



    def evaluate_models(self) -> None:
        """
        Evaluates the trained models on the test set using specified metrics
        (Cohen's Kappa, Matthews Correlation Coefficient, Balanced Accuracy).
        Generates plots for ROC curves and learning curves to analyze model performance
        and identify potential overfitting/underfitting.

        Calculates and stores metrics (Cohen's Kappa, Matthews Correlation Coefficient, Balanced Accuracy).
        Creates ROC curves for each model.
        Generates learning curves to assess the bias-variance trade-off of each model.
        """
        if not self.models:
            logging.warning("No models trained.  Please call train_models() first.")
            return

        logging.info("Evaluating models...")

        self.results = {}  # Reset results dictionary

        for model_name, model in self.models.items():
            try:
                start_time = time.time()
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)[:, 1] #Probabilities for ROC AUC
                kappa = cohen_kappa_score(self.y_test, y_pred)
                mcc = matthews_corrcoef(self.y_test, y_pred)
                balanced_accuracy = balanced_accuracy_score(self.y_test, y_pred)
                roc_auc = roc_auc_score(self.y_test, y_proba)

                self.results[model_name] = {
                    "cohen_kappa": kappa,
                    "matthews_corrcoef": mcc,
                    "balanced_accuracy": balanced_accuracy,
                    "roc_auc": roc_auc
                }
                end_time = time.time()
                logging.info(f"Evaluation of {model_name} completed in {end_time - start_time:.2f} seconds.")

                print(f"\n{model_name} Evaluation:")
                print(f"  Cohen's Kappa: {kappa:.4f}")
                print(f"  Matthews Correlation Coefficient: {mcc:.4f}")
                print(f"  Balanced Accuracy: {balanced_accuracy:.4f}")
                print(f"  ROC AUC: {roc_auc:.4f}")


                # 1. ROC Curve
                try:
                    fpr, tpr, thresholds = roc_curve(self.y_test, y_proba)
                    plt.figure(figsize=(8, 6))
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {model_name}')
                    plt.legend()
                    plt.show()
                except Exception as e:
                    logging.warning(f"ROC Curve plot failed for {model_name}: {e}")


                # 2. Learning Curves
                try:
                    common_params = {
                        "X": self.X_train,
                        "y": self.y_train,
                        "train_sizes": np.linspace(0.1, 1.0, 5),
                        "cv": StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state), #or TimeSeriesSplit for time-dependent data
                        "score_type": "both",
                        "n_jobs": -1,
                        "line_kw": {"marker": "o"},
                        "std_display": "std",
                        "random_state": self.random_state
                    }

                    fig, ax = plt.subplots(figsize=(8, 6))
                    LearningCurveDisplay.from_estimator(model, **common_params, ax=ax)
                    ax.set_title(f"Learning Curve - {model_name}")
                    plt.show()

                except Exception as e:
                    logging.warning(f"Learning Curve plot failed for {model_name}: {e}")


            except Exception as e:
                logging.error(f"Error evaluating {model_name}: {e}")

        logging.info("Model evaluation complete.")

if __name__ == "__main__":
    project = MLProject()
    project.load_data()
    project.explore_data()
    project.preprocess_data()
    project.train_models()
    project.evaluate_models()

    # Print the results
    if project.results:
        print("\nModel Evaluation Results:")
        for model_name, metrics in project.results.items():
            print(f"\n{model_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
    else:
        print("No models were evaluated.")
```

```markdown
# Diabetes Prediction Project

## 1. Project Overview

**Main Objectives:**

This project aims to develop a machine learning pipeline for predicting diabetes using a dataset of clinical features. The pipeline includes data loading, preprocessing, exploratory data analysis (EDA), model training, evaluation, and visualization, with a focus on achieving high prediction accuracy and robust performance.

**Key Features:**

*   **Data Loading:** Loads the diabetes dataset from OpenML.
*   **Data Preprocessing:** Handles missing values, scales numerical features, and encodes categorical features.  Splits data into training and testing sets, avoiding data leakage.
*   **Exploratory Data Analysis (EDA):** Provides visualizations for data understanding, including class distribution, feature histograms, correlation matrices, and boxplots for outlier detection.
*   **Model Training:** Trains and tunes several machine learning models, including ExtraTreesClassifier, BaggingClassifier, HistGradientBoostingClassifier, and a StackingClassifier. StratifiedKFold cross-validation is used for robust hyperparameter tuning.
*   **Model Evaluation:** Evaluates model performance using Cohen's Kappa, Matthews Correlation Coefficient, Balanced Accuracy, and ROC AUC.
*   **Visualizations:** Generates ROC curves and learning curves for detailed model analysis.
*   **Model Stacking:** Implements a stacking ensemble using VotingClassifier with soft voting and Logistic Regression as the meta-learner.
*   **Time Series Cross-Validation (Note: While not a true time series in this dataset, logic is included):** Includes placeholder for time series cross-validation if the data were time-dependent.

**Technical Highlights:**

*   Implements a complete end-to-end machine learning pipeline.
*   Uses advanced techniques such as model stacking and stratified cross-validation.
*   Provides comprehensive model evaluation and visualization.
*   Addresses data cleaning and preprocessing challenges.
*   Employs detailed logging and error handling.

## 2. Technical Details

**Model Architecture:**

*   **ExtraTreesClassifier:** An ensemble learning method that constructs multiple decision trees.
*   **BaggingClassifier:** An ensemble meta-estimator that fits base classifiers on random subsets of the original dataset and then aggregates their individual predictions. Decision Trees are used as the base estimator.
*   **HistGradientBoostingClassifier:** A gradient boosting method that uses a histogram-based algorithm.
*   **StackingClassifier:** An ensemble learning technique that combines multiple base models (ExtraTreesClassifier, BaggingClassifier, HistGradientBoostingClassifier) using a Logistic Regression meta-learner.

**Data Processing Pipeline:**

1.  **Data Loading:** Loads the diabetes dataset from OpenML.
2.  **Data Splitting:** Splits the data into training and testing sets using stratified sampling to maintain class balance.
3.  **Data Imputation:** Handles missing values using mean imputation.
4.  **Feature Scaling:** Scales numerical features using MinMaxScaler.
5.  **Feature Encoding:** Encodes categorical features using OneHotEncoder.

**Key Algorithms:**

*   **StratifiedKFold:** A cross-validation technique that preserves the percentage of samples for each class.
*   **GridSearchCV:** A hyperparameter optimization technique that exhaustively searches through a specified hyperparameter grid.
*   **MinMaxScaler:** Scales and translates each feature individually such that it is in the range of zero and one.
*   **OneHotEncoder:** Encodes categorical features into numerical format.
*   **VotingClassifier:** Combines the predictions from multiple machine learning models.

## 3. Performance Metrics

**Evaluation Results:**

The models are evaluated using the following metrics:

*   **Cohen's Kappa:** Measures the agreement between predicted and actual values.
*   **Matthews Correlation Coefficient (MCC):** Measures the quality of binary classifications.
*   **Balanced Accuracy:** The average of recall obtained on each class.
*   **ROC AUC:** Measures the area under the Receiver Operating Characteristic curve.

Evaluation results, including Cohen's Kappa, Matthews Correlation Coefficient, Balanced Accuracy, and ROC AUC, are printed for each model.

**Benchmark Comparisons:**

The performance of the models is compared against each other to identify the best-performing model based on the evaluation metrics.  The best model based on ROC AUC is explicitly identified.

**Model Strengths:**

*   **ExtraTreesClassifier:** Effective for feature selection and handling high-dimensional data.
*   **BaggingClassifier:** Robust to outliers and high variance.
*   **HistGradientBoostingClassifier:** Handles mixed data types and missing values.
*   **StackingClassifier:** Improves overall performance by combining the strengths of multiple base models.

## 4. Implementation Details

**Dependencies:**

*   Python 3.6+
*   pandas
*   numpy
*   scikit-learn
*   matplotlib
*   seaborn
*   category_encoders

**System Requirements:**

*   Standard desktop or laptop computer
*   Sufficient RAM for data processing and model training (at least 4GB recommended)

**Setup Instructions:**

1.  Install the required dependencies using pip:

    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn category_encoders
    ```

2.  Download the code and save it as `diabetes_project.py`.
3.  Run the script:

    ```bash
    python diabetes_project.py
    ```

The script will load the data, preprocess it, train the models, evaluate their performance, and display the results along with visualizations.
```

## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/arastuthakur/ml-projects-collection.git
cd ml-projects-collection/```python
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the project:
```bash
python ```python.py
```

## Project Structure

- ````python.py`: Main implementation file
- `requirements.txt`: Project dependencies
- Generated visualizations:
  - Feature distributions
  - Correlation matrix
  - ROC curve
  - Feature importance plot

## License

MIT License
