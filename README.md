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

## Project Structure

```
Assignment 2/
├── app.py                 # Streamlit web app for model comparison and predictions
├── train.py               # Training script: trains all 6 models and saves them
├── notebook.ipynb         # Jupyter notebook for EDA, cleaning, and experimentation
├── bank.csv               # Main bank marketing dataset used for training (root)
├── train.csv              # Train split / sample data (if generated)
├── test.csv               # Test split / sample data (if generated)
├── model_comparison.csv   # Evaluation metrics for all models (from train.py)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── model/                 # Saved models (created by train.py)
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    └── label_encoders.pkl
```

| Item | Purpose |
|------|--------|
| **`app.py`** | Streamlit application to compare model metrics, view confusion matrices, and run predictions on uploaded CSV data. |
| **`train.py`** | Script to train all six classifiers on `bank.csv`, evaluate them, save metrics to `model_comparison.csv`, and persist models in `model/`. |
| **`notebook.ipynb`** | Interactive notebook for data cleaning, exploratory data analysis (EDA), and model exploration. |
| **`model/`** | Directory where serialized models (`.pkl`) and label encoders are stored after running `train.py`. |
| **`model_comparison.csv`** | CSV of evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC, TP/FP/FN/TN) for each model. |

---

## c. Models Used: Performance Comparison

Make a comparison table with the evaluation metrics calculated for all the 6 models as below (values from `model_comparison.csv`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.797 | 0.873 | 0.796 | 0.769 | 0.782 | 0.593 |
| **Decision Tree** | 0.807 | 0.873 | 0.762 | 0.862 | 0.809 | 0.620 |
| **kNN** | 0.790 | 0.852 | 0.797 | 0.748 | 0.771 | 0.578 |
| **Naive Bayes** | 0.751 | 0.811 | 0.722 | 0.774 | 0.747 | 0.505 |
| **Random Forest (Ensemble)** | 0.856 | 0.916 | 0.826 | 0.881 | 0.853 | 0.713 |
| **XGBoost (Ensemble)** | 0.857 | 0.926 | 0.832 | 0.875 | 0.853 | 0.715 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | This model works well as a basic starting model (baseline). It gives balanced Precision and Recall (around 0.77–0.80). Its AUC (0.873) is lower than ensemble models, but it is simple and easy to understand. Since it is not very complex, it usually does not overfit easily. Good choice when we want interpretability. |
| **Decision Tree** | This model gives high Recall (0.862), which means it is good at identifying positive cases. AUC is 0.873. Because the tree depth is controlled, it avoids too much overfitting. It performs better than simple models in some cases but may still have some bias–variance trade-off. |
| **kNN** | This model gives moderate performance (Accuracy 0.790, AUC 0.852). It depends heavily on scaling of data (that’s why StandardScaler is used). It can be sensitive to noisy data and high dimensions. Overall, it performs okay but is not the best model in this project. |
| **Naive Bayes** | This model has the lowest Accuracy (0.751) and MCC (0.505). It assumes that all features are independent, which is often not true in real-world data. Because of this strong assumption, performance is lower. However, it does not overfit and works fast. |
| **Random Forest (Ensemble)** | This model performs very well across all metrics (for example, F1 = 0.853 and AUC = 0.916). Since it combines many decision trees, it reduces overfitting and improves stability. It gives strong and reliable performance. |
| **XGBoost (Ensemble)** | This is the best performing model in the project. It has the highest Accuracy (0.857), AUC (0.926), Precision (0.832), and MCC (0.715). It improves performance by combining weak models step by step (boosting). It controls overfitting using parameters like depth and subsampling. Best choice if we want highest performance. |


---

## How to Train the Models

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the training script** (from the project root). The script reads `bank.csv`, trains all six models, saves metrics to `model_comparison.csv`, and writes serialized models into the `model/` folder:

   ```bash
   python train.py
   ```

   Ensure `bank.csv` is present in the project root before running. After training, the `model/` directory and `model_comparison.csv` will be created/updated.

---

## Run Streamlit App

Start the Streamlit app from the project root:

```bash
streamlit run app.py
```

The app opens in your browser at **http://localhost:8501** (or the URL shown in the terminal).

- **Comparison of Model:** View the metrics table from `model_comparison.csv`, select a model to see detailed metrics and confusion matrix, and compare two models side by side.
- **Predict:** Upload a CSV file with the same feature columns as the training data (e.g. `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `contact`, `day`, `month`, `duration`, `campaign`, `pdays`, `previous`, `poutcome`). Choose a model, click **Make Predictions**, and view the predicted labels and optional download.
- **Dataset Info:** Explore dataset overview and statistics.

**Note:** Train the models first with `python train.py` so that `model/` and `model_comparison.csv` exist; otherwise the app will show warnings about missing models or metrics.

⚠️ **App may take a few seconds to wake up due to free hosting.**

---

## How to Run (Notebook)

### 1. Navigate to the Project Directory

```bash
cd bank_marketing_campaign_ml_classification
```

(or your project folder path)

### 2. Install Dependencies

Make sure you have **Python** installed. Then run:

```bash
pip install -r requirements.txt
```

### 3. Run the Analysis in Jupyter

Open the Jupyter Notebook to view the data cleaning, EDA, and model training:

```bash
jupyter notebook
```

Then open `notebook.ipynb` from your browser.

### 4. Clone the repository (optional)

```bash
git clone https://github.com/kartik-powar-3103/bank_marketing_campaign_ml_classification.git
```
