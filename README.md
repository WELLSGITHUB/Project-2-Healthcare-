# Project-2-Healthcare-
![alt text](Resources/dataset-cover.jpg)
# Stroke Prediction Model

This project demonstrates how to use machine learning techniques for predicting the likelihood of stroke occurrence in patients using a dataset of healthcare information. The dataset includes various features such as age, hypertension, heart disease, smoking status, BMI, and more. We preprocess the data, handle missing values, encode categorical variables, and use different models to predict stroke outcomes.


## Table of Contents
1. [Import Libraries](#import-libraries)
2. [Data Preparation](#data-preparation)
3. [Resolving Missing Data and Rescaling](#resolving-missing-data-and-rescaling)
4. [Categorical Data Encoded](#categorical-data-encoded)
5. [Balancing the Data](#balancing-the-data)
6. [Modeling](#modeling)
    - Random Forest Classifier
    - Logistic Regression
8. [Conclusion](#conclusion)

## Import Libraries

We begin by importing necessary Python libraries for data changes, modeling, and evaluation.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from imblearn.over_sampling import SMOTE
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import GridSearchCV


## Data Preparation

### The dataset includes the following columns:

1)  id: unique identifier
2)  gender: "Male", "Female" or "Other"
3)  age: age of the patient
4)  hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
5)  heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
6)  ever_married: "No" or "Yes"
7)  work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
8)  Residence_type: "Rural" or "Urban"
9)  avg_glucose_level: average glucose level in blood
10) bmi: body mass index
11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
12) ==stroke: 1 if the patient had a stroke or 0 if not==
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient.`

### Installation of Data

To run this project, you'll need Python 3 and the following libraries:

- `pandas`
- `matplotlib`
- `numpy`
- `sklearn`

You can install the required packages using pip:
pip install pandas numpy matplotlib scikit-learn imbalanced-learn statsmodels
```bash
pip install pandas matplotlib numpy sklearn
```
### Data Preprocessing:
- The dataset is loaded and cleaned by handling missing values and incorrect entries (e.g., dropping rows with missing BMI or irrelevant gender data).
- Categorical variables such as `gender`, `ever_married`, `work_type`, and `Residence_type` are encoded.
- Numerical features such as `age`, `bmi`, and `avg_glucose_level` are scaled.
- The dataset is loaded using `pd.read_csv`:

```python
stroke_df = pd.read_csv("Resources/healthcare-dataset-stroke-data.csv")
```
- Then, any rows with missing values (except for bmi) are dropped, and rows with "Other" in the gender column are removed.

### Data Splitting:
- The dataset is split into training and testing sets using `train_test_split`.
- Imbalanced data is handled by techniques like Random Oversampling and SMOTE (Synthetic Minority Over-sampling Technique).
## Resolving Missing Data and Rescaling

The missing values in `smoking_status` are handled by assuming that individuals under 18 years old have never smoked. The data is then scaled using `StandardScaler`:

```python
scaler = StandardScaler()
col_scale = ['age', 'bmi', 'avg_glucose_level']
X_train[col_scale] = scaler.fit_transform(X_train[col_scale])
X_test[col_scale] = scaler.transform(X_test[col_scale])
```
## Categorical Data Encoded

Categorical variables are encoded using `OneHotEncoder` and `OrdinalEncoder` for ordinal variables (such as `smoking_status`).

```python
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type']
X_train_encoded = pd.get_dummies(X_train, columns=categorical_columns, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_columns, drop_first=True)

smoke_ord_enc = OrdinalEncoder(categories=[['smokes', 'formerly smoked', 'never smoked']], 
                               encoded_missing_value=-1, handle_unknown='use_encoded_value', unknown_value=-1)
X_train_encoded['smoking_status_ordinal'] = smoke_ord_enc.fit_transform(X_train_encoded['smoking_status'].values.reshape(-1, 1))
X_test_encoded['smoking_status_ordinal'] = smoke_ord_enc.transform(X_test_encoded['smoking_status'].values.reshape(-1, 1))
```
## Balancing the Data

Imbalanced classes are handled using SMOTE:

```python
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_encoded, y_train)
```


## Modeling:
The project applies multiple classification algorithms such as:
- Random Forest
- Logistic Regression
- Decision Trees
- Support Vector Machines (SVM)

A `GridSearchCV` is used to fine-tune the logistic regression model's hyperparameters.

### Evaluation:
- The models are evaluated using metrics like precision, recall, F1-score, and balanced accuracy to ensure reliable performance, especially in the presence of imbalanced data. 

## Conclusion
- This project demonstrates how data preprocessing techniques, such as handling missing values, feature scaling, encoding categorical variables, and oversampling, can help in building a model to predict strokes. By using multiple classifiers and resampling techniques, we aim to create a balanced model capable of identifying at-risk patients.

## Data Sources

- Stroke Data sourced from (https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).


## Acknowledgements

[Accreditation](https://www.kaggle.com/fedesoriano)
(Confidential Source) - Use only for educational purposes
If you use this dataset in your research, please credit the author.
License: Data files © Original Authors




