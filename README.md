# Project-2-Healthcare-
![alt text](Resources/dataset-cover.jpg)
# Stroke Prediction Model

This project demonstrates how to use machine learning techniques for predicting the likelihood of stroke occurrence in patients using a dataset of healthcare information. The dataset includes various features such as age, hypertension, heart disease, smoking status, BMI, and more. We preprocess the data, handle missing values, encode categorical variables, and use different models to predict stroke outcomes.

[title](https://www.example.com)

## Table of Contents
1. [Import Libraries](#import-libraries)
2. [Data Preparation](#data-preparation)
3. [Data Scaling](#data-scaling)
4. [Data Encoding](#data-encoding)
5. [Handling Class Imbalance](#handling-class-imbalance)
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

## The dataset includes the following columns:

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

## Installation of Data

To run this project, you'll need Python 3 and the following libraries:

- `pandas`
- `matplotlib`
- `numpy`
- `sklearn`

You can install the required packages using pip:
pip install pandas matlib numpy sklearn

```bash
pip install pandas matplotlib numpy sklearn
```
## Handling Missing Data

We address the missing data in the following ways:

- **BMI**: We drop rows where the BMI is missing, as the missing percentage is low.
- **Gender**: We drop rows where the gender is labeled as "Other" (only one row contains this label).
- **Smoking Status**: We fill missing `smoking_status` values, assuming that most "Unknown" statuses for individuals under 18 years of age are "never smoked".
 
## Get to Know

| Link  | Description  |
|:------|:-------------|
| [Cardiovasculares](https://www.who.int/health-topics/cardiovascular-diseases#tab=tab_1) ||| World Health Organization. |||
| [Heart Disease Prediction](https://www.kaggle.com/code/farzadnekouei/heart-disease-prediction/notebook) ||| kaggle.com |||
| [Trends In Heart Disease](https://catalog.data.gov/dataset/rates-an diseasd-trends-in-heart-disease-and-stroke-mortality-among-us-adults-35-by-county-a-2000-45659 ||| data.gov. |||
| [STROKE)](https://www.who.int/southeastasia/news/detail/28-10-2021-world-stroke-day) ||| World Health Organization. |||

## Usage

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Pnwankwo2/Project-2-Healthcare-.git
    cd Project-2-Healthcare-
    ```

2. **Place the CSV files** (`healthcare-dataset-stroke-data`) in the `resources` directory.



3. **View the generated visualizations** that illustrate trends and correlations.

4. **Presentation** A copy of the presentation Google Docs can be found in the images folder.

## Data Sources

- Stroke Data sourced from (https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).



## Analysis

The analysis is performed in several key steps:

1. **Data Import**: The script reads the CSV files into Pandas DataFrames.
2. **Data Cleaning**: Dates are formatted appropriately, and columns are renamed for clarity.
3. **Data Merging**: Wastewater and case data are combined into a single DataFrame for comparative analysis.
4. **Visualization**: Various plots are generated to illustrate trends over time and regional differences.

| Syntax | Description |
| ----------- | ----------- |
| Header | Title |
| Paragraph | Text |

[title](https://www.kaggle.com/fedesoriano)


### Key Metrics Analyzed

- **Antigens in Wastewater**: Concentration of virus particles found in wastewater samples.
- **COVID-19 Cases**: Rolling average cases per 100,000 residents.


Check the license (commons etc. which one is it) and credit the author
## Acknowledgements
(Confidential Source) - Use only for educational purposes
If you use this dataset in your research, please credit the author.
License: Data files © Original Authors




