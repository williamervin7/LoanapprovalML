# 🏦 Loan Approval Prediction – Week 7 Project

This project focuses on predicting whether a loan application will be approved based on applicant information. The goal was to build an interpretable model with solid performance, especially in identifying cases that should be **not approved**.


## 📁 Dataset

The dataset includes information about loan applicants, such as:

- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area
- Education
- Marital Status
- Gender  
- **Target**: `Loan_Status` (Approved = 1, Not Approved = 0)


## 🧹 Data Cleaning & Feature Engineering

- Converted `Loan_Status` to binary numeric format
- Created new feature: `TotalIncome = ApplicantIncome + CoapplicantIncome`
- Used `ColumnTransformer` and `Pipeline` for preprocessing
- Imputed missing values
- Standardized numerical features
- One-hot encoded categorical variables


## 🔍 Model Selection

Evaluated multiple models using cross-validation (CV accuracy):

| Model               | CV Accuracy |
|--------------------|-------------|
| Random Forest       | 0.7394      |
| Gradient Boosting   | 0.7867      |
| XGBoost             | 0.7655      |
| Logistic Regression | **0.8127**  |
| LGBMClassifier      | 0.7801      |

✅ **Logistic Regression** was selected due to highest CV score and interpretability.


## ⚖️ Addressing Class Imbalance

Initial results showed low recall for Class 0 (Not Approved), meaning many false negatives. This is problematic for loan approvals, where wrongly approving a risky applicant can lead to financial loss.

To improve recall for Class 0, we added:


class_weight='balanced'
This helped redistribute model focus to underrepresented classes.

## 🔧 Hyperparameter Tuning
Used GridSearchCV with 5-fold CV to tune the logistic regression model.
```python
param_grid = {
    'logisticregression__C': [0.01, 0.1, 1, 10, 100],
    'logisticregression__penalty': ['l2'],
    'logisticregression__solver': ['lbfgs', 'liblinear'],
    'logisticregression__class_weight': ['balanced']
}
```
Best Parameters:

* `C=10`
* `solver='liblinear'
* `class_weight='balanced'`
## 📊 Final Model Results
Train Set:

* Accuracy: 0.80
* Class 0 Recall: 0.45
* Class 1 Recall: 0.96

Test Set:

* Accuracy: 0.80

* Class 0 Recall: 0.42

* Class 1 Recall: 0.98

## 🎯 Threshold Tuning
Plotted the precision-recall curve and adjusted threshold to optimize trade-offs:

Custom threshold of 0.52 balanced Class 0 recall and precision better than the default.
This was a strategic adjustment to minimize false approvals without sacrificing overall performance.

## 🔍 Feature Importance
Extracted and visualized top features using model coefficients:
Most impactful features:
* Credit History
* Total Income
* Loan Amount
* Education
* Marital Status

## ✂️ Feature Reduction
Rebuilt the model using only top features to reduce complexity:

| Metric    | Full Model | Reduced Model |
|-----------|------------|----------------|
| Accuracy  | 0.80       | 0.80           |
| Precision | 0.79       | 0.79           |
| Recall    | 0.94       | 0.94           |
| F1 Score  | 0.86       | 0.86           |


✅ Result: The reduced model performed just as well with fewer features.

## 📌 Key Takeaways
* Logistic Regression performed best among all models tested

* Addressed class imbalance using class_weight='balanced'

* Improved Class 0 recall with threshold tuning

* Feature selection helped simplify the model with no performance loss

## 🛠️ Tools Used
* Python

* Pandas & NumPy

* Scikit-learn (Pipelines, GridSearchCV, Metrics)

* Matplotlib

* Seaborn

## 🚀 Next Steps
* Export model for deployment

* Test on new (production-like) data
## ✅ Completion
This wraps up my Week 7 Data Science Project.