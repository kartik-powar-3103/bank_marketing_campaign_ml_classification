# Bank Marketing Campaign - Machine Learning Classification

This project aims to predict whether a client will subscribe to a term deposit based on marketing campaign data from a Portuguese banking institution.

## a. Problem Statement
The objective is to build a classification model to predict the success of a marketing campaign (variable `y`). By identifying the characteristics of customers who are most likely to subscribe to a term deposit, the bank can optimize its marketing resources and increase conversion rates.

## b. Dataset Description
The dataset is the "Bank Marketing" dataset from the UCI Machine Learning Repository. It contains 41,188 entries and 20 input features, categorized into:
* **Customer Demographics:** Age, job, marital status, education.
* **Financial Status:** Default status, housing loan, personal loan.
* **Campaign Contact Details:** Contact type, month, day of the week, duration.
* **Socio-economic Indicators:** Employment variation rate, consumer price index, euribor 3 month rate.

---

## c. Models Used: Performance Comparison

The following table summarizes the evaluation metrics for the six models implemented in this project:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.910 | 0.93 | 0.67 | 0.39 | 0.49 | 0.47 |
| **Decision Tree** | 0.892 | 0.71 | 0.52 | 0.52 | 0.52 | 0.46 |
| **kNN** | 0.902 | 0.86 | 0.61 | 0.40 | 0.48 | 0.44 |
| **Naive Bayes** | 0.819 | 0.81 | 0.35 | 0.61 | 0.44 | 0.36 |
| **Random Forest (Ensemble)** | 0.915 | 0.94 | 0.68 | 0.48 | 0.56 | 0.53 |
| **XGBoost (Ensemble)** | 0.916 | 0.95 | 0.67 | 0.51 | 0.58 | 0.54 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Solid baseline accuracy but relatively low recall, meaning it misses several potential subscribers. |
| **Decision Tree** | Shows a balanced recall and precision but suffers from lower AUC compared to ensemble methods. |
| **kNN** | Performs well in terms of accuracy but is outperformed by tree-based models on this specific dataset. |
| **Naive Bayes** | Highest recall (0.61) among individual models but lowest precision, leading to many false positives. |
| **Random Forest (Ensemble)** | Greatly improves the F1-score and stability by aggregating multiple decision trees. |
| **XGBoost (Ensemble)** | **Best Overall Performer.** Achieved the highest AUC (0.95) and F1-score (0.58), making it the most reliable for deployment. |

---

## How to Run
## 🚀 How to Run

### 1️⃣ Navigate to the Project Directory

```bash
cd bank_marketing_campaign_ml_classification
```

### 2️⃣ Install Dependencies

Make sure you have **Python** installed. Then run:

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Analysis

Open the Jupyter Notebook to view the data cleaning, EDA, and model training:

```bash
jupyter notebook
```

Then open the relevant `.ipynb` file from your browser.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/kartik-powar-3103/bank_marketing_campaign_ml_classification.git](https://github.com/kartik-powar-3103/bank_marketing_campaign_ml_classification.git)
