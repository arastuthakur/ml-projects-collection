#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Machine Learning Project - Regression With Feature Engineering And With Cross-Validation And With Model Interpretation And With Smote Balancing

This module implements a complete machine learning pipeline for regression with feature engineering and with cross-validation and with model interpretation and with SMOTE balancing
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
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score
)from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor,
    ExtraTreesRegressor, BaggingRegressor, HistGradientBoostingRegressor
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
except ImportError:
    logger.warning("Some boosting libraries not available. Using default regressors.")

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
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.svm import SVR
from sklearn.utils import resample
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class MLProject:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}

    def load_data(self):
        try:
            self.data = fetch_openml('house_prices', version=1)
            logger.info("Data loaded successfully.")
        except Exception as e:
            logger.error("Error loading data: {}".format(e))

    def explore_data(self):
        # Implement EDA with visualizations specific to with feature engineering, with cross-validation, with model interpretation, with SMOTE balancing

    def preprocess_data(self):
        # Implement preprocessing specific to with feature engineering, with cross-validation, with model interpretation, with SMOTE balancing
        categorical_features = ['LotArea', 'YearBuilt']
        numerical_features = ['1stFlrSF', '2ndFlrSF', 'FullBath', 'HalfBath']

        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numerical_transformer = StandardScaler()
        polynomial_transformer = PolynomialFeatures(degree=2)

        preprocessor = ColumnTransformer(transformers=[
            ('categorical', categorical_transformer, categorical_features),
            ('numerical', numerical_transformer, numerical_features),
            ('polynomial', polynomial_transformer, numerical_features)
        ])

        X_preprocessed = preprocessor.fit_transform(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_preprocessed, self.data['SalePrice'], test_size=0.2, random_state=42)

        logger.info("Data preprocessed successfully.")

    def train_models(self):
        # Train and tune multiple models (RandomForestRegressor, GradientBoostingRegressor, SVR)

        models = [
            ('RandomForestRegressor', RandomForestRegressor()),
            ('GradientBoostingRegressor', GradientBoostingRegressor()),
            ('SVR', SVR())
        ]

        for name, model in models:
            # Tune hyperparameters using GridSearchCV
            params = {'max_depth': [3, 5, 7], 'n_estimators': [100, 200, 300]}
            grid_search = GridSearchCV(model, params, cv=5, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)

            # Store tuned model
            self.models[name] = grid_search.best_estimator_

            logger.info("Model {} trained successfully.".format(name))

    def evaluate_models(self):
        # Calculate metrics (mean_squared_log_error, mean_poisson_deviance, mean_gamma_deviance) and create plots

        metrics = {
            'mean_squared_log_error': mean_squared_log_error,
            'mean_poisson_deviance': mean_poisson_deviance,
            'mean_gamma_deviance': mean_gamma_deviance
        }

        for name, model in self.models.items():
            self.results[name] = {}
            for metric_name, metric in metrics.items():
                score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring=metric)
                self.results[name][metric_name] = score.mean()

            logger.info("Model {} evaluated successfully.".format(name))

    def run(self):
        self.load_data()
        self.explore_data()
        self.preprocess_data()
        self.train_models()
        self.evaluate_models()

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

def main():
    try:
        # Initialize the MLProject class
        project = MLProject()

        # Run the complete pipeline
        project.run_pipeline()

        # Print evaluation results for all models
        for model_name, model in project.models.items():
            print(f"Evaluation results for {model_name}:")
            print(f"Mean squared error: {mean_squared_error(project.y_test, model.predict(project.X_test))}")
            print(f"R2 score: {r2_score(project.y_test, model.predict(project.X_test))}")

        # Compare model performances
        print("Model comparison:")
        print(pd.DataFrame({
            "Model": list(project.models.keys()),
            "Mean squared error": [mean_squared_error(project.y_test, model.predict(project.X_test)) for model in project.models.values()],
            "R2 score": [r2_score(project.y_test, model.predict(project.X_test)) for model in project.models.values()]
        }))

        # Feature engineering analysis
        print("Feature engineering analysis:")
        print(pd.DataFrame({
            "Feature": project.X_train.columns,
            "Importance": permutation_importance(project.models["Random Forest"], project.X_train, project.y_train).importances_mean
        }).sort_values("Importance", ascending=False))

        # Cross-validation analysis
        print("Cross-validation analysis:")
        print(pd.DataFrame({
            "Model": list(project.models.keys()),
            "Cross-validation score": [np.mean(cross_val_score(model, project.X_train, project.y_train, cv=5)) for model in project.models.values()]
        }))

        # Model interpretation analysis
        print("Model interpretation analysis:")
        print(pd.DataFrame({
            "Model": list(project.models.keys()),
            "Most important feature": [permutation_importance(model, project.X_train, project.y_train).importances_mean.argmax() for model in project.models.values()]
        }))

        # SMOTE balancing analysis
        print("SMOTE balancing analysis:")
        print(pd.DataFrame({
            "Model": list(project.models.keys()),
            "SMOTE balanced score": [mean_squared_error(project.y_test, model.predict(SMOTE().fit_resample(project.X_test, project.y_test)[0])) for model in project.models.values()]
        }))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()