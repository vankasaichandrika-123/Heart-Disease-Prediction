# ❤️  Heart-Disease-Prediction

##ML Heart Disease Prediction Project

#1. Project Overview

This project is a machine-learning classification pipeline for predicting the target class from a heart-disease dataset.

The current pipeline performs:

Dataset loading

Null-value checking

Train/test split

Yeo-Johnson transformation

Constant-feature removal

Quasi-constant-feature removal

Pearson correlation / hypothesis-based feature selection

Training-data class balancing

Model training

Model evaluation

ROC curve visualization

The project also writes execution details into separate log files under the logs directory.

#2. Dataset Information

The current run contains:

Rows: 303

Columns: 14

Input features before feature selection: 13

Target column: target

Missing values: 0

The logs show a training set of 242 records and a test set of 61 records.

After balancing, the training data contains:

Class 0: 133

Class 1: 133

Total: 266

Source: main.log.

#3. Feature Engineering and Selection**

Initial features

The initial feature set contains:

age
sex
cp
trestbps
chol
fbs
restecg
thalach
exang
oldpeak
slope
ca
thal

After Yeo-Johnson transformation, the feature names become:

age_yeo_trim
sex_yeo_trim
cp_yeo_trim
trestbps_yeo_trim
chol_yeo_trim
fbs_yeo_trim
restecg_yeo_trim
thalach_yeo_trim
exang_yeo_trim
oldpeak_yeo_trim
slope_yeo_trim
ca_yeo_trim
thal_yeo_trim

The feature-selection log shows:

Constant-feature removal

Removed:

fbs_yeo_trim

Remaining features: 12

Quasi-constant-feature removal

Removed:

trestbps_yeo_trim
chol_yeo_trim
exang_yeo_trim
ca_yeo_trim

Remaining features: 8

Hypothesis testing

The logged p-values resulted in:

restecg_yeo_trim

being removed.

Final model input features:

age_yeo_trim
sex_yeo_trim
cp_yeo_trim
thalach_yeo_trim
oldpeak_yeo_trim
slope_yeo_trim
thal_yeo_trim

Source: fs.log.

#4. Models Used

The project evaluates:

K-Nearest Neighbors (KNN)

Gaussian Naive Bayes

Logistic Regression

Decision Tree

Random Forest

AdaBoost

Gradient Boosting

XGBoost

The current implementation also creates an ROC curve visualization for these models.

#5. Current Test Results

The following values are from the current all_models.log run.

Model

Test Accuracy

KNN

57.38%

#Naive Bayes

#80.33%

Logistic Regression

59.02%

Decision Tree

60.66%

Random Forest

50.82%

AdaBoost

59.02%

Gradient Boosting

62.30%

XGBoost

55.74%

The current logs therefore show different performance across the tested models. These are test-set measurements from this particular run and should not be interpreted as general clinical performance.

#6. Naive Bayes Result

The current Naive Bayes test result contains:

Accuracy: 0.8032786885245902

Confusion matrix:

[[25  4]
 [ 8 24]]

Classification report:

              precision    recall  f1-score   support

0                0.76       0.86      0.81        29
1                0.86       0.75      0.80        32

accuracy                              0.80        61
macro avg          0.81       0.81      0.80        61
weighted avg       0.81       0.80      0.80        61

#7. ROC Curve

The project currently generates an ROC visualization containing curves for:

KNN
LR
NB
DT
RF
ADA
GB
XGB

Important implementation note

The current implementation uses model predict() outputs when calling roc_curve(). For a standard ROC-AUC evaluation, probability scores or decision scores should normally be used instead of hard 0/1 predictions.

A future improvement is to use:

predict_proba(X_test)[:, 1]

for models that support probability prediction, or an appropriate decision score for models that do not.

The current graph should therefore be treated as the visualization produced by the existing implementation, not as a complete set of validated ROC-AUC values.

#8. Logging

Logs are stored in:

D:\ML_Projects\ML_HEART_project\logs

Current log files:

main.log
fs.log
yeo_timing.log
all_models.log

main.log

Contains:

Dataset shape

Null-value information

Train/test sizes

Class distribution

Balancing results

Main model output

fs.log

Contains:

Feature names before/after transformation

Constant-feature removal

Quasi-constant-feature removal

p-values

Hypothesis-based feature removal

yeo_timing.log

Contains:

Feature names before transformation

Feature names after Yeo-Johnson transformation

all_models.log

Contains:

Individual model sections

Accuracy

Confusion matrix

Classification report

ROC/AUC stage information

#9. Project Structure

A recommended structure is:

ML_HEART_project/
│
├── main.py
├── all_models.py
├── fs.py
├── yeo_timing.py
├── log_code.py
├── index.html
├── README.md
│
├── data/
│   └── heart dataset
│
├── logs/
│   ├── main.log
│   ├── fs.log
│   ├── yeo_timing.log
│   └── all_models.log
│
└── templates/
    └── index.html

If Flask is being used, the HTML file should normally be placed under:

templates/index.html

#10. Running the Project

From the project directory:

cd D:\ML_Projects\ML_HEART_project
python main.py

The machine-learning pipeline should then generate/update the log files.

#11. Prediction UI

The included index.html provides a simple web form for the final seven model-input features:

age_yeo_trim
sex_yeo_trim
cp_yeo_trim
thalach_yeo_trim
oldpeak_yeo_trim
slope_yeo_trim
thal_yeo_trim

These are the final transformed features recorded by the feature-selection pipeline. The prediction backend must apply the same preprocessing assumptions used during model training.

Do not enter raw clinical values into these fields unless the backend explicitly performs the required Yeo-Johnson transformation before prediction.

#12. Important Medical Disclaimer

This is a machine-learning project for technical/educational use. A model prediction is not a medical diagnosis and should not be used by itself to make healthcare decisions.

For a production medical application, the model would require appropriate validation, calibration, clinical evaluation, data governance, and deployment controls.
