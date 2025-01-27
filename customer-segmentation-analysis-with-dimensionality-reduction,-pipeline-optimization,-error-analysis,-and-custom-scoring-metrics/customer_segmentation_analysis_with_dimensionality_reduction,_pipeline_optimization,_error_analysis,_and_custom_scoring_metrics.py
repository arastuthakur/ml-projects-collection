#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Machine Learning Project - Customer Segmentation Analysis With With Dimensionality Reduction And With Pipeline Optimization And With Error Analysis And With Custom Scoring Metrics

This module implements a complete machine learning pipeline for Customer Segmentation Analysis with with dimensionality reduction and with pipeline optimization and with error analysis and with custom scoring metrics
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
)from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN,
    SpectralClustering, Birch, MiniBatchKMeans,
    MeanShift, AffinityPropagation, OPTICS
)

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
import logging
from sklearn.cluster import MeanShift, AffinityPropagation, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score
from sklearn.decomposition import PCA

class MLProject:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.data = None
        self.models = []
        self.results = {}

    def load_data(self):
        try:
            self.data = pd.read_csv("wholesale-customers.csv")
            self.logger.info("Data loaded successfully.")
        except Exception as e:
            self.logger.error("Error loading data.", e)

    def explore_data(self):
        try:
            # EDA specific to dimensionality reduction
            # ...

            # EDA specific to pipeline optimization
            # ...

            # EDA specific to error analysis
            # ...

            # EDA specific to custom scoring metrics
            # ...
            self.logger.info("Data exploration complete.")
        except Exception as e:
            self.logger.error("Error exploring data.", e)

    def preprocess_data(self):
        try:
            # Preprocessing specific to dimensionality reduction
            # ...

            # Preprocessing specific to pipeline optimization
            # ...

            # Preprocessing specific to error analysis
            # ...

            # Preprocessing specific to custom scoring metrics
            # ...
            self.logger.info("Data preprocessing complete.")
        except Exception as e:
            self.logger.error("Error preprocessing data.", e)

    def train_models(self):
        try:
            # Initialize and train multiple models
            models = [
                MeanShift(),
                AffinityPropagation(),
                OPTICS()
            ]

            # Define parameters for grid search
            parameters = {
                'MeanShift': {
                    'bandwidth': [0.1, 0.2, 0.3]
                },
                'AffinityPropagation': {
                    'damping': [0.5, 0.7, 0.9]
                },
                'OPTICS': {
                    'min_samples': [5, 10, 15]
                }
            }

            # Create pipelines combining preprocessing and models
            pipelines = [
                Pipeline([
                    ('preprocessing', ColumnTransformer([('scaling', StandardScaler(), ['Age', 'Income']]())),
                    ('model', MeanShift())
                ]),
                Pipeline([
                    ('preprocessing', ColumnTransformer([('scaling', StandardScaler(), ['Age', 'Income']]())),
                    ('model', AffinityPropagation())
                ]),
                Pipeline([
                    ('preprocessing', ColumnTransformer([('scaling', StandardScaler(), ['Age', 'Income']]())),
                    ('model', OPTICS())
                ])
            ]

            # Perform grid search and store results
            for pipeline, parameters in zip(pipelines, parameters.values()):
                grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=5)
                grid_search.fit(self.data.drop('Channel', axis=1), self.data['Channel'])

                # Store model and results
                self.models.append(grid_search.best_estimator_)
                self.results[pipeline.named_steps['model'].__class__.__name__] = grid_search.cv_results_

            self.logger.info("Models trained successfully.")
        except Exception as e:
            self.logger.error("Error training models.", e)

    def evaluate_models(self):
        try:
            # Calculate metrics for each model
            for model in self.models:
                y_true = self.data['Channel']
                y_pred = model.predict(self.data.drop('Channel', axis=1))

                # Calculate adjusted_rand, adjusted_mutual_info, homogeneity
                adjusted_rand = adjusted_rand_score(y_true, y_pred)
                adjusted_mutual_info = adjusted_mutual_info_score(y_true, y_pred)
                homogeneity = homogeneity_score(y_true, y_pred)

                # Store results
                self.results[model.__class__.__name__]['adjusted_rand'] = adjusted_rand
                self.results[model.__class__.__name__]['adjusted_mutual_info'] = adjusted_mutual_info
                self.results[model.__class__.__name__]['homogeneity'] = homogeneity

            # Create plots
            # ...

            self.logger.info("Models evaluated successfully.")
        except Exception as e:
            self.logger.error("Error evaluating models.", e)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Define custom scoring metrics
def specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)

def negative_predictive_value(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn)

# Main function
def main():
    try:
        # Initialize MLProject class
        project = MLProject()
        
        # Load and preprocess data
        data = pd.read_csv('customer_segmentation.csv')
        data = data.dropna()
        X = data.drop('Label', axis=1)
        y = data['Label']
        
        # Dimensionality reduction
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
        
        # Create pipeline with grid search
        grid_params = {
            'model__C': [0.1, 1, 10],
            'model__max_depth': [2, 5, 10]
        }
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
        grid_search = GridSearchCV(pipeline, grid_params, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        # Evaluate models
        models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier()]
        scores = []
        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            specificity_score = specificity(y_test, y_pred)
            negative_predictive_value_score = negative_predictive_value(y_test, y_pred)
            scores.append({
                'Model': type(model).__name__,
                'Accuracy': accuracy,
                'F1 Score': f1,
                'Specificity': specificity_score,
                'Negative Predictive Value': negative_predictive_value_score
            })
        
        # Compare model performances
        scores_df = pd.DataFrame(scores).set_index('Model')
        print(scores_df)
        
        # Plot model performances
        scores_df.drop(['Specificity', 'Negative Predictive Value'], axis=1).plot.bar()
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparision')
        plt.show()
    
    except Exception as e:
        print(f'An error occurred: {e}')

# Run main function
if __name__ == '__main__':
    main()