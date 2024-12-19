# Amex-Credit-Default-Prediction
**Amex Default Prediction**
Predicting customer default risks using advanced machine learning techniques. This project was developed as part of the **American Express - Default Prediction Kaggle Competition.**

**Project Overview** 

Credit default prediction is a crucial challenge for financial institutions. This project leverages machine learning to predict the likelihood of default using transaction and behavioral data. The pipeline includes data preprocessing, feature engineering, model training, evaluation using a custom metric, and interpretability analysis.

# **Project Workflow** 
**Problem Statement:**

- Predict customer default risks using a large, tabular dataset.
- Evaluate performance with the Amex Metric, a custom evaluation metric.


**Data Exploration:**

- Investigated dataset structure and feature types.
- Addressed missing values and outliers.
- Visualized feature distributions and relationships.

**Feature Engineering:**

- Created new features such as **ratios**, **rolling statistics**, and **lag features**.
- Aggregated customer-level information (**mean, standard deviation**, etc.).
- Encoded categorical variables and scaled numerical features.

**Model Building:**

- Implemented **XGBoost** as the primary model.
- Experimented with **hyperparameter tuning** using **Optuna**.
- Added ensemble models (**LightGBM, Random Forest**) for comparison.

**Evaluation:**

- Used **ROC-AUC** and the competition’s **Amex Metric** for validation.
- Applied **cross-validation** for robust performance evaluation.

**Interpretability:**
Utilized **SHAP** to analyze feature importance and understand model decisions.

# Key Features
**Data Handling:**

- Processed large datasets in chunks for memory efficiency.
- Handled missing values using imputation techniques.
- Generated interaction features and time-based rolling statistics.

**Model Development:**

- Trained models using **XGBoost, LightGBM** and **Random Forest**.
- Tuned hyperparameters with **Optuna** to optimize performance.

**Model Evaluation:**

- Implemented the **Amex Metric** for custom evaluation.
- Visualized feature contributions using **SHAP summary plots**.

# Technologies Used

**Programming Language:** Python

**Libraries:**
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** XGBoost, LightGBM, Scikit-Learn
- **Model Interpretability:** SHAP
- **Hyperparameter Tuning:** Optuna
**Tools:** Jupyter Notebook, GitHub, Kaggle

# Getting Started
**1. Clone the Repository**

git clone https://github.com/AYUSHKUMARSAHOO04/amex-default-prediction.git

cd amex-default-prediction

**2. Install Dependencies**

pip install -r requirements.txt

**3. Download the Dataset**

You have two options to download the dataset:

**Option 1: Download Manually**

Visit the **Kaggle Competition** page.

**Download** the files:
- train_data.csv

- test_data.csv

- train_labels.csv

Place them in the data/ directory of this repository.

**Option 2: Use Kaggle API**

Ensure the **Kaggle API** is set up on your system:

Download **kaggle.json** from your Kaggle account settings.

Place it in the ~/.kaggle/ directory (Linux/Mac) or %USERPROFILE%/.kaggle/ (Windows).

Run the following script to download the dataset:

python download_data.py

**4. Run the Project**

Use the provided notebooks or scripts to preprocess data, train models, and generate predictions.

# Key Insights:
- Features like **balance_to_income_ratio** and **spend_to_balance_ratio** were highly predictive.
- Model interpretability using SHAP revealed critical factors influencing default predictions.

**Future Work**
- Incorporate **deep learning models** like **TabNet** for tabular data.
- Experiment with **advanced ensemble techniques** (***stacking** and **blending**).
- Deploy the model as an API for **real-time inference**.
