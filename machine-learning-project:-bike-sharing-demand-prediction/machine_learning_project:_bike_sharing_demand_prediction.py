#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Machine Learning Project - Bike Sharing Demand Prediction With With Feature Selection And With Pipeline Optimization And With Model Interpretation And With Smote Balancing

This module implements a complete machine learning pipeline for Bike Sharing Demand Prediction with with feature selection and with pipeline optimization and with model interpretation and with SMOTE balancing
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
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

class MLProject:
    def __init__(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize project variables
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}

    def load_data(self):
        """Load fetch_openml('bike_sharing', version=1) dataset"""
        try:
            self.logger.info("Loading data...")
            self.data = fetch_openml('bike_sharing', version=1)
            self.logger.info("Data loaded successfully.")
        except Exception as e:
            self.logger.error("Error loading data.", exc_info=True)
            raise e

    def explore_data(self):
        """Perform EDA with visualizations specific to with feature selection, with pipeline optimization, with model interpretation, with SMOTE balancing"""
        try:
            self.logger.info("Exploring data...")
            # Check for null values
            self.logger.info("Checking for null values...")
            null_counts = self.data.isnull().sum()
            null_counts[null_counts > 0]
            # Check for duplicates
            self.logger.info("Checking for duplicates...")
            duplicate_counts = self.data.duplicated().sum()
            duplicate_counts[duplicate_counts > 0]
            # Check for outliers
            self.logger.info("Checking for outliers...")
            for col in self.data.columns:
                sns.boxplot(x=self.data[col])
                plt.show()
            # Check for correlations
            self.logger.info("Checking for correlations...")
            corr_matrix = self.data.corr()
            sns.heatmap(corr_matrix, annot=True)
            plt.show()
            self.logger.info("Data exploration complete.")
        except Exception as e:
            self.logger.error("Error exploring data.", exc_info=True)
            raise e

    def preprocess_data(self):
        """Implement preprocessing specific to with feature selection, with pipeline optimization, with model interpretation, with SMOTE balancing"""
        try:
            self.logger.info("Preprocessing data...")

            # Create feature selection pipeline
            feature_selection_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_regression, k=5))
            ])

            # Create preprocessing pipeline
            preprocessing_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('onehotencoder', OneHotEncoder(sparse=False)),
                ('feature_selection', feature_selection_pipeline)
            ])

            # Apply preprocessing pipeline to data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.drop('count', axis=1), self.data['count'], test_size=0.2, random_state=42)
            self.X_train = preprocessing_pipeline.fit_transform(self.X_train)
            self.X_test = preprocessing_pipeline.transform(self.X_test)
            self.logger.info("Data preprocessing complete.")
        except Exception as e:
            self.logger.error("Error preprocessing data.", exc_info=True)
            raise e

    def train_models(self):
        """Train and tune multiple models (XGBRegressor, LGBMRegressor, CatBoostRegressor)"""
        try:
            self.logger.info("Training models...")

            # Create models
            self.models['XGBRegressor'] = xgb.XGBRegressor()
            self.models['LGBMRegressor'] = lgb.LGBMRegressor()
            self.models['CatBoostRegressor'] = cb.CatBoostRegressor()

            # Create parameter grids for each model
            param_grids = {
                'XGBRegressor': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15],
                    'learning_rate': [0.1, 0.01, 0.001]
                },
                'LGBMRegressor': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15],
                    'learning_rate': [0.1, 0.01, 0.001]
                },
                'CatBoostRegressor': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 10, 15],
                    'learning_rate': [0.1, 0.01, 0.001]
                }
            }

            # Perform grid search for each model
            for model_name, model in self.models.items():
                self.logger.info(f"Tuning {model_name}...")
                grid_search = GridSearchCV(model, param_grids[model_name], cv=5, n_jobs=-1)
                grid_search.fit(self.X_train, self.y_train)
                self.models[model_name] = grid_search.best_estimator_

            self.logger.info("Models trained successfully.")
        except Exception as e:
            self.logger.error("Error training models.", exc_info=True)
            raise e

    def evaluate_models(self):
        """Calculate metrics (mse, mae, r2) and create plots"""
        try:
            self.logger.info("Evaluating models...")

            # Evaluate each model
            for model_name, model in self.models.items():
                y_pred = model.predict(self.X_test)
                mse = mean_squared_error(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                r2 = r2_score(self.y_test, y_pred)
                self.results[model_name] = {'mse': mse, 'mae': mae, 'r2': r2}

            # Create table of results
            results_df = pd.DataFrame(self.results).T
            results_df.index.rename('Model', inplace=True)
            print(results_df)

            # Create plot of results
            plt.figure(figsize=(10, 6))
            plt.plot(results_df.index, results_df['mse'], label='MSE')
            plt.plot(results_df.index, results_df['mae'], label='MAE')
            plt.plot(results_df.index, results_df['r2'], label='R2')
            plt.legend()
            plt.show()
            self.logger.info("Models evaluated successfully.")
        except Exception as e:
            self.logger.error("Error evaluating models.", exc_info=True)
            raise e

if __name__ == '__main__':
    # Create project object
    project = MLProject()

    # Load data
    project.load_data()

    # Explore data
    project.explore_data()

    # Preprocess data
    project.preprocess_data()

    # Train models
    project.train_models()

    # Evaluate models
    project.evaluate_models()

def main():
    try:
        # Initialize the MLProject class
        project = MLProject()

        # Run the complete pipeline
        project.run_pipeline()

        # Prints evaluation results for all models
        print("Evaluation results for all models:")
        for model_name, model in project.models.items():
            print(f"{model_name}:")
            print(model.evaluate())

        # Compares model performances
        print("Comparison of model performances:")
        project.compare_models()

        # Includes proper error handling
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()