# Project-2-Healthcare-

Check the license (commons etc. which one is it) and credit the author

The following are the given attributes for the dataset:
Column 1), 2) etc.:
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
12) stroke: 1 if the patient had a stroke or 0 if not
*Note: "Unknown" in smoking_status means that the information is unavailable for this patient.

Handle Missing Values - we used the fillna() function to fill the missing values in the bmi column with the bmi mean.

Standardize Data formats - we changed the following categorical columns to category data type using the astype('category') function: gender, ever_married, work_type, Residence_type, smoking_status.  

Consistency - the id column is in integer format and for consistency we changed it to as string format using the astype function.
