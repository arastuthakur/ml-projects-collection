#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Machine Learning Project - Binary Classification

This module implements a complete machine learning pipeline for binary classification
using various algorithms and comprehensive evaluation metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import logging
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class MLProject:

    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}

    def load_data(self, path: str):
        """Loads the load_breast_cancer dataset.

        Args:
            path (str): Path to the dataset.
        """
        try:
            self.data = pd.read_csv(path)
            logging.info("Dataset loaded successfully.")
        except FileNotFoundError:
            logging.error("Dataset file not found at the specified path.")

    def explore_data(self):
        """Performs EDA with visualizations.
        """
        sns.pairplot(self.data)
        plt.show()
        logging.info("EDA completed.")

    def preprocess_data(self, test_size: float = 0.25):
        """Splits and scales data.

        Args:
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.25.
        """
        try:
            # Split data into train and test sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.drop('target', axis=1), 
                                                                                    self.data['target'], 
                                                                                    test_size=test_size, 
                                                                                    random_state=42)

            # Scale data
            scaler = StandardScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

            logging.info("Data preprocessed successfully.")
        except KeyError:
            logging.error("Target column 'target' not found in the dataset.")

    def train_models(self):
        """Trains and tunes multiple models (RandomForestClassifier, GradientBoostingClassifier, SVC).
        """
        # Define models
        models = {
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'SVC': SVC()
        }

        # Define hyperparameter tuning params
        param_grids = {
            'RandomForestClassifier': {'n_estimators': [100, 200, 500], 'max_depth': [5, 10, 20]},
            'GradientBoostingClassifier': {'n_estimators': [100, 200, 500], 'max_depth': [5, 10, 20]},
            'SVC': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        }

        # Train and tune models
        for name, model in models.items():
            try:
                grid = GridSearchCV(model, param_grids[name], cv=5)
                grid.fit(self.X_train, self.y_train)

                self.models[name] = grid.best_estimator_
                logging.info(f"Model {name} trained successfully.")
            except ValueError:
                logging.error(f"Error training model {name}.")

    def evaluate_models(self):
        """Calculates metrics (accuracy, precision, recall, f1, roc_auc) and creates plots.
        """
        results = {}

        # Evaluate models
        for name, model in self.models.items():
            try:
                y_pred = model.predict(self.X_test)

                results[name] = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred),
                    'recall': recall_score(self.y_test, y_pred),
                    'f1': f1_score(self.y_test, y_pred),
                    'roc_auc': roc_auc_score(self.y_test, y_pred)
                }

                logging.info(f"Model {name} evaluated successfully.")
            except ValueError:
                logging.error(f"Error evaluating model {name}.")

        # Create boxplots
        df_results = pd.DataFrame(results).T
        df_results.boxplot()
        plt.xlabel('Model')
        plt.ylabel('Metric Value')
        plt.show()

        self.results = results

    def compare_models(self):
        """Compares model results.

        Returns:
            The best model based on the highest roc_auc score.
        """
        best_model = max(self.results, key=lambda x: self.results[x]['roc_auc'])
        logging.info(f"The best model is {best_model} with an ROC AUC score of {self.results[best_model]['roc_auc']}.")
        return best_model

if __name__ == '__main__':
    project = MLProject()
    project.load_data('breast_cancer.csv')
    project.explore_data()
    project.preprocess_data()
    project.train_models()
    project.evaluate_models()
    best_model = project.compare_models()

def main():
    try:
        # Initialize the MLProject class
        project = MLProject(
            config_file="config.yaml",
            data_dir="data",
            results_dir="results",
        )

        # Run the complete pipeline
        project.run_pipeline()

        # Print evaluation results for all models
        for model in project.models:
            print(f"Evaluation results for {model.name}:")
            print(model.metrics)

        # Compare model performances
        project.compare_models()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()