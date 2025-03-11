# Mall Customer Clustering

# Mall Customer Clustering

This project implements a machine learning pipeline for clustering mall customers based on their spending behavior and annual income. It addresses key challenges in data analysis, including missing value imputation, hyperparameter optimization, model comparison, and robust evaluation using multiple metrics.

## 1. Project Overview

**Objectives:**

* Segment mall customers into distinct groups based on their purchasing habits.
* Identify optimal clustering models and hyperparameters for effective segmentation.
* Provide insights into customer behavior to inform targeted marketing strategies.

**Key Features:**

* Automated data preprocessing, including missing value imputation and feature scaling.
* Hyperparameter tuning using `GridSearchCV` for optimal model performance.
* Evaluation using multiple clustering metrics (Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index).
* Comparative analysis of different clustering algorithms (KMeans, DBSCAN, Agglomerative Clustering).
* Visualization of clustering results for intuitive interpretation.

**Technical Highlights:**

* Robust error handling and informative logging.
* Efficient implementation using Scikit-learn and Pandas.
* Modular design for easy customization and extension.

## 2. Technical Details

**Model Architectures:**

* **KMeans:** Partitions data into *k* clusters by minimizing the within-cluster sum of squares.
* **DBSCAN:** Density-based clustering that groups data points based on their density and identifies outliers.
* **Agglomerative Clustering:**  A hierarchical clustering method that builds clusters by iteratively merging or splitting them.

**Data Processing Pipeline:**

1. **Data Loading:** Reads data from a CSV file.
2. **Missing Value Imputation:**  Imputes missing values using the mean of each feature.
3. **Feature Scaling:** Standardizes features using `StandardScaler` to ensure equal weighting during model training.

**Key Algorithms:**

* `SimpleImputer` for missing value imputation.
* `StandardScaler` for feature scaling.
* `GridSearchCV` for hyperparameter optimization.
* `silhouette_score`, `calinski_harabasz_score`, `davies_bouldin_score` for clustering evaluation.


## 3. Performance Metrics

**Evaluation Results:**

The performance of each model is evaluated using the following metrics:

* **Silhouette Score:** Measures how similar a data point is to its own cluster compared to other clusters (higher is better).
* **Calinski-Harabasz Index:** Measures the ratio of between-cluster variance to within-cluster variance (higher is better).
* **Davies-Bouldin Index:** Measures the average similarity between each cluster and its most similar cluster (lower is better).

Specific results and comparisons are printed to the console after training and evaluation.

**Model Strengths:**

The best performing model is determined based on the chosen evaluation metrics and is available for further analysis or deployment.  Model comparison provides insights into which algorithm is most suitable for the given data.


## 4. Implementation Details

**Dependencies:**

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

**System Requirements:**

Standard Python environment with the above-mentioned libraries installed.

**Setup Instructions:**

1. Clone the repository.
2. Install dependencies:  `pip install -r requirements.txt` (create a `requirements.txt` file listing the libraries).
3. Run the main script: `python main.py`
4. Replace `'Mall_Customers.csv'` in `main.py` with the path to your dataset.


This README provides a comprehensive overview of the Mall Customer Clustering project. The code is designed to be robust, flexible, and informative, enabling effective customer segmentation and analysis.  Customize hyperparameter grids, visualizations, and the model comparison logic as needed to best suit your data and analytical objectives.


## Setup and Installation

1. Clone this repository:
```bash
git clone https://github.com/arastuthakur/ml-projects-collection.git
cd ml-projects-collection/mall-customer-clustering
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the project:
```bash
python mall_customer_clustering.py
```

## Project Structure

- `mall_customer_clustering.py`: Main implementation file
- `requirements.txt`: Project dependencies
- Generated visualizations:
  - Feature distributions
  - Correlation matrix
  - ROC curve
  - Feature importance plot

## License

MIT License
