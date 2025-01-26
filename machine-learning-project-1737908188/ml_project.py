#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Machine Learning Project - Classification With Feature Selection And With Ensemble Methods And With Error Analysis And With Custom Scoring Metrics

This module implements a complete machine learning pipeline for classification with feature selection and with ensemble methods and with error analysis and with custom scoring metrics
using various algorithms and comprehensive evaluation metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
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

import logging
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MLProject:
    def __init__(self) -> None:
        self.df = None
        self.models = {}
        self.results = {}

    def load_data(self) -> None:
        """Load the fetch_openml('diabetes', version=1) dataset."""
        try:
            self.df = pd.read_csv("diabetes.csv")
        except FileNotFoundError:
            logging.error("Data file not found.")

    def explore_data(self) -> None:
        """Perform EDA with visualizations specific to with feature selection, with ensemble methods, with error analysis, with custom scoring metrics."""
        sns.pairplot(self.df)
        plt.show()

        # Feature selection
        chi2_selector = SelectKBest(chi2, k=5)
        chi2_selector.fit(self.df.drop("target", axis=1), self.df["target"])
        selected_features = self.df.drop("target", axis=1).columns[chi2_selector.get_support()]
        print("Selected features:", selected_features)

        # Ensemble methods
        estimators = [
            ("lr", LogisticRegression()),
            ("ridge", RidgeClassifier()),
            ("sgd", SGDClassifier()),
        ]
        ensemble_model = VotingClassifier(estimators=estimators)

        # Error analysis
        y_train, y_test = train_test_split(self.df["target"], test_size=0.25)
        ensemble_model.fit(self.df.drop("target", axis=1), y_train)
        y_pred = ensemble_model.predict(self.df.drop("target", axis=1))
        print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

        # Custom scoring metrics
        kappa = cohen_kappa_score(y_test, y_pred)
        print("Cohen's kappa:", kappa)

    def preprocess_data(self) -> None:
        """Implement preprocessing specific to with feature selection, with ensemble methods, with error analysis, with custom scoring metrics."""
        # Feature selection
        self.df = self.df.loc[:, ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6", "target"]]

        # Ensemble methods
        # No preprocessing necessary

        # Error analysis
        # No preprocessing necessary

        # Custom scoring metrics
        # No preprocessing necessary

    def train_models(self) -> None:
        """Train and tune multiple models (LogisticRegression, RidgeClassifier, SGDClassifier)."""
        # Feature selection
        feature_selection = Pipeline(
            steps=[("scaler", StandardScaler()), ("selector", SelectKBest(chi2, k=5))]
        )

        # Ensemble methods
        ensemble_model = VotingClassifier(estimators=[("lr", LogisticRegression()), ("ridge", RidgeClassifier()), ("sgd", SGDClassifier())])

        # Error analysis
        # No specific preprocessing or model required

        # Custom scoring metrics
        # No specific preprocessing or model required

        models = [
            ("lr", LogisticRegression()),
            ("ridge", RidgeClassifier()),
            ("sgd", SGDClassifier()),
            ("feature_selection", feature_selection),
            ("ensemble", ensemble_model),
        ]

        for name, model in models:
            parameters = {
                "lr__C": [0.001, 0.01, 0.1, 1],
                "ridge__alpha": [0.001, 0.01, 0.1, 1],
                "sgd__alpha": [0.001, 0.01, 0.1, 1],
                "feature_selection__selector__k": [5, 10, 15, 20],
                "ensemble__voting": ["hard", "soft"],
            }
            grid_search = GridSearchCV(model, parameters, cv=5)
            grid_search.fit(self.df.drop("target", axis=1), self.df["target"])
            self.models[name] = grid_search.best_estimator_

    def evaluate_models(self) -> None:
        """Calculate metrics (cohen_kappa, matthews_corrcoef, balanced_accuracy) and create plots."""
        y_train, y_test = train_test_split(self.df["target"], test_size=0.25)
        results = {}

        for name, model in self.models.items():
            model.fit(self.df.drop("target", axis=1), y_train)
            y_pred = model.predict(self.df.drop("target", axis=1))

            kappa = cohen_kappa_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

            results[name] = (kappa, mcc, balanced_accuracy)

        self.results = pd.DataFrame(results, index=["Cohen's kappa", "Matthews corrcoef", "Balanced accuracy"]).T

        # Create plots
        self.results.plot(kind="bar", title="Model evaluation results")
        plt.show()


if __name__ == "__main__":
    project = MLProject()
    project.load_data()
    project.explore_data()
    project.preprocess_data()
    project.train_models()
    project.evaluate_models()

def main():
    """
    Main function to run the ML project
    """

    try:
        # Initialize the MLProject class
        project = MLProject(
            data_path="data/train.csv",
            target_variable="target_variable",
            features_to_select=["feature1", "feature2", "feature3"],
            ensemble_models=["RandomForestClassifier", "AdaBoostClassifier"],
            scoring_metrics=["accuracy", "f1_score", "recall_score", "precision_score"],
            custom_scoring_metrics=[MyCustomScoringMetric()],
            error_analysis_metrics=["confusion_matrix", "classification_report"],
            visualizations=["bar_chart", "line_chart"],
        )

        # Run the complete pipeline
        project.run()

        # Print evaluation results for all models
        for model_name, results in project.results.items():
            print(f"Evaluation results for {model_name}:")
            for metric_name, metric_value in results.items():
                print(f"\t{metric_name}: {metric_value}")

        # Compare model performances
        project.compare_models()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()