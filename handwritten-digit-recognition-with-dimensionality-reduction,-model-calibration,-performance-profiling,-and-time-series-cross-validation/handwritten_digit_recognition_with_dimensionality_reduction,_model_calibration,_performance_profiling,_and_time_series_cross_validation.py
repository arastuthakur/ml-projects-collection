#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Machine Learning Project - Handwritten Digit Recognition With With Dimensionality Reduction And With Model Calibration And With Performance Profiling And With Time Series Cross-Validation

This module implements a complete machine learning pipeline for Handwritten Digit Recognition with with dimensionality reduction and with model calibration and with performance profiling and with time series cross-validation
using various algorithms and comprehensive evaluation metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier, BaggingClassifier, HistGradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
except ImportError:
    logger.warning("Some boosting libraries not available. Using default classifiers.")

import logging
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_recall_auc_score, log_loss

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class MLProject:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.models = {}
        self.scores = {}

    def load_data(self):
        logging.info("Loading dataset...")
        digits = fetch_openml('mnist_784', version=1, return_X_y=True)
        self.X_train, self.y_train = digits[0], digits[1]
        self.X_test, self.y_test = digits[2], digits[3]

    def explore_data(self):
        logging.info("Exploring data...")
        # TODO: Implement EDA specific to with dimensionality reduction, with model calibration, with performance profiling, with time series cross-validation

    def preprocess_data(self):
        logging.info("Preprocessing data...")
        # TODO: Implement preprocessing specific to with dimensionality reduction, with model calibration, with performance profiling, with time series cross-validation

    def train_models(self):
        logging.info("Training models...")
        models = [KNeighborsClassifier(), DecisionTreeClassifier(), AdaBoostClassifier()]
        for model in models:
            pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
            pipeline.fit(self.X_train, self.y_train)
            self.models[model.__class__.__name__] = pipeline

    def evaluate_models(self):
        logging.info("Evaluating models...")
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            self.scores[model_name] = {
                'roc_auc': roc_auc_score(self.y_test, y_pred),
                'precision_recall_auc': precision_recall_auc_score(self.y_test, y_pred),
                'log_loss': log_loss(self.y_test, y_pred)
            }

    def plot_results(self):
        logging.info("Plotting results...")
        sns.barplot(x=list(self.scores.keys()), y=[score['roc_auc'] for score in self.scores.values()])
        plt.xlabel('Model')
        plt.ylabel('ROC AUC')
        plt.title('ROC AUC Scores')
        plt.show()

if __name__ == '__main__':
    project = MLProject()
    project.load_data()
    project.explore_data()
    project.preprocess_data()
    project.train_models()
    project.evaluate_models()
    project.plot_results()

import mlproject
from mlproject.evaluation import evaluate_models
from mlproject.utils import print_error


def main():
    try:
        # Initialize MLProject
        project = mlproject.MLProject()

        # Run the complete pipeline
        project.run_pipeline()

        # Print evaluation results for all models
        results = evaluate_models(project.models)
        print(results)

        # Compare model performances
        print("Best model:", max(results, key=lambda x: x['accuracy']))

    except Exception as e:
        print_error(e)


if __name__ == "__main__":
    main()