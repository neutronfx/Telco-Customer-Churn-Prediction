# Telco-Customer-Churn-Prediction
Predicting customer churn for a telecom company using Python, Pandas, Scikit-learn, and XGBoost.

## Introduction/Objective

This project aims to predict customer churn for a fictional telecom company based on historical customer data. Customer churn, the rate at which customers stop doing business with a company, is a critical metric. By identifying customers likely to churn, the company can proactively engage them with targeted retention strategies, potentially reducing revenue loss and improving customer satisfaction. The objective is to build and evaluate machine learning models to classify customers as likely to churn ('Yes') or not ('No').

## Data Source

The dataset used for this analysis is the "Telco Customer Churn" dataset, publicly available on Kaggle.
*   **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
*   **Dataset:** `WA_Fn-UseC_-Telco-Customer-Churn.csv` (Note: Data file not included in this repository, please download from source).

## Methodology

The project followed these key steps:

1.  **Data Loading & Initial Inspection:** Loaded the dataset using Pandas and performed initial checks on data types, missing values, and basic statistics.
2.  **Data Cleaning & Preprocessing:**
    *   Handled missing values in the `TotalCharges` column (converted to numeric, imputed missing values using the median).
    *   Dropped the irrelevant `customerID` column.
3.  **Exploratory Data Analysis (EDA):** Analyzed relationships between various features and the target variable (`Churn`) using visualizations (count plots, box plots, KDE plots) with Matplotlib and Seaborn. Key insights were gathered regarding customer demographics, account information (tenure, contract, charges), and service subscriptions.
4.  **Feature Engineering:**
    *   Encoded binary categorical features (including the target variable 'Churn') using Scikit-learn's `LabelEncoder`.
    *   Encoded multi-category nominal features using Pandas `get_dummies` (One-Hot Encoding), dropping the first category to avoid multicollinearity.
    *   Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using Scikit-learn's `StandardScaler`.
5.  **Model Building:** Trained three different classification models:
    *   Logistic Regression (with `class_weight='balanced'`)
    *   Random Forest Classifier (with `class_weight='balanced_subsample'`)
    *   XGBoost Classifier (with `scale_pos_weight` calculated to handle imbalance)
6.  **Model Evaluation:** Evaluated models using a held-out test set (20% of data). Key metrics included Accuracy, Precision, Recall, F1-Score (specifically for the 'Churn' class), and the Confusion Matrix. Stratified splitting was used to maintain class distribution.
7.  **Feature Importance Analysis:** Extracted and visualized feature importances from the Random Forest and XGBoost models to understand key drivers of churn.

## Tools/Libraries Used

*   **Python 3.x**
*   **Pandas:** Data manipulation and analysis.
*   **NumPy:** Numerical operations.
*   **Matplotlib & Seaborn:** Data visualization.
*   **Scikit-learn:** Data preprocessing (LabelEncoder, StandardScaler, train_test_split), model building (LogisticRegression, RandomForestClassifier), and evaluation (accuracy_score, confusion_matrix, classification_report).
*   **XGBoost:** Model building (XGBClassifier).
*   **Jupyter Notebook:** Interactive development environment.

## Exploratory Data Analysis Highlights

EDA revealed several factors strongly correlated with churn:
*   **Tenure:** Customers with shorter tenure are significantly more likely to churn.
*   **Contract Type:** Month-to-month contracts have a much higher churn rate compared to one or two-year contracts.
*   **Monthly Charges:** Higher monthly charges are associated with increased churn likelihood.
*   **Key Services:** Lack of value-added services like Online Security and Tech Support correlates with higher churn rates.
*   **Demographics:** Senior citizens showed a higher churn propensity, while customers with partners or dependents showed lower churn rates.

## Modeling and Evaluation

Three models were trained and evaluated. Performance on the test set for predicting churn (Class '1'):

| Model               | Accuracy | Churn Recall | Churn Precision | Churn F1-Score | Notes                                                    |
| :------------------ | :------- | :----------- | :-------------- | :------------- | :------------------------------------------------------- |
| Logistic Regression | 74%      | **78%**      | 51%             | **0.61**       | Best recall, good F1, but lowest precision.              |
| Random Forest       | **78%**  | 47%          | **62%**         | 0.53           | Highest accuracy & precision, but poor recall.          |
| XGBoost             | 76%      | 67%          | 55%             | 0.60           | Good balance between recall & precision, good accuracy. |

**Conclusion on Models:** While Logistic Regression achieved the highest Recall (finding the most churners), and Random Forest the highest Precision (being most accurate when predicting churn), **XGBoost offered the best overall balance** between Recall (67%) and Precision (55%), resulting in a solid F1-score (0.60). Depending on business priorities (minimize missed churners vs. minimize wasted retention efforts), either Logistic Regression or XGBoost could be preferred.

## Key Findings & Feature Importance

Based on the XGBoost model's feature importance analysis, the top drivers influencing customer churn prediction include:

1.  **Contract Type (Two year, One year):** Longer contracts strongly indicate lower churn risk.
2.  **Internet Service (Fiber optic, No):** Fiber optic associated with churn (potentially due to cost/complexity), no internet service indicates very low churn risk within this context.
3.  **Tenure:** Lower tenure remains a key predictor of higher churn likelihood.
4.  **Payment Method (Electronic check):** Often associated with higher churn in this dataset.
5.  **Monthly Charges / Total Charges:** Higher costs contribute to churn risk.
6.  **Key Services (Tech Support, Online Security, etc.):** Lack of these services increases churn likelihood.

*(Note: Feature importance list based on XGBoost results from our analysis)*

## Conclusion & Potential Next Steps

This project successfully developed models to predict telecom customer churn. The analysis identified key factors like contract duration, tenure, monthly charges, and specific services as significant drivers. The XGBoost model provided a good balance in identifying potential churners while maintaining reasonable precision.

**Potential Next Steps:**

*   **Hyperparameter Tuning:** Further optimize XGBoost (or other models) using techniques like GridSearchCV or RandomizedSearchCV.
*   **Advanced Feature Engineering:** Create interaction terms or ratio features to potentially improve model performance.
*   **Alternative Imbalance Handling:** Experiment with resampling techniques like SMOTE.
*   **Deployment:** Develop an API or simple application to serve the model for real-time predictions (outside the scope of this notebook).

## How to Run

1.  Ensure you have Python and the necessary libraries installed (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost).
2.  Download the dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) from the [Kaggle link](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) provided above and place it in the same directory as the notebook.
3.  Open and run the Jupyter Notebook (`Telco_Churn_Analysis.ipynb` or your notebook name) cells sequentially.
