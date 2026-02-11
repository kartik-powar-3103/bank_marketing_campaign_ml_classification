# Core
import numpy as np
import pandas as pd

# Visualization (optional)
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

data_df = pd.read_csv("bank.csv")
data_df.shape

data_df = pd.read_csv("bank.csv")
dataset_source = "Kaggle (https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data)" 
n_samples = data_df.shape[0]
n_features = data_df.shape[1]

data_df["deposit"] = data_df["deposit"].map({"yes": 1, "no": 0})

target = "deposit"

features = list(data_df.columns)
features.remove(target)

X = data_df[features]
y = data_df[target]

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

categorical_cols = data_df.select_dtypes(include=["object", "category"]).columns.tolist()

from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Probability needed for AUC
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)


    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return metrics

log_reg_pl = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

log_reg_pl.fit(X_train, y_train)
lgr_metrics = evaluate_model(log_reg_pl, X_test, y_test)
print(lgr_metrics)

dt = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

dt.fit(X_train, y_train)
dt_metrics = evaluate_model(dt, X_test, y_test)
print(dt_metrics)

knn_pl = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(n_neighbors=5))
])

knn_pl.fit(X_train, y_train)
knn_metrics = evaluate_model(knn_pl, X_test, y_test)
print(knn_metrics)

nb = GaussianNB()

nb.fit(X_train, y_train)
nb_metrics = evaluate_model(nb, X_test, y_test)
print(nb_metrics)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)
rf_metrics = evaluate_model(rf, X_test, y_test)
print(rf_metrics)

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)

xgb.fit(X_train, y_train)
xgb_metrics = evaluate_model(xgb, X_test, y_test)
print(xgb_metrics)

results = pd.DataFrame.from_dict({
    "Logistic Regression": lgr_metrics,
    "Decision Tree": dt_metrics,
    "KNN": knn_metrics,
    "Naive Bayes": nb_metrics,
    "Random Forest": rf_metrics,
    "XGBoost": xgb_metrics
}, orient="index")

print(results)

results.to_csv("model_comparison.csv")

import joblib
import os

os.makedirs("model", exist_ok=True)

joblib.dump(label_encoders, "model/label_encoders.pkl")
joblib.dump(log_reg_pl, "model/logistic_regression.pkl")
joblib.dump(dt, "model/decision_tree.pkl")
joblib.dump(knn_pl, "model/knn.pkl")
joblib.dump(nb, "model/naive_bayes.pkl")
joblib.dump(rf, "model/random_forest.pkl")
joblib.dump(xgb, "model/xgboost.pkl")

test_df = pd.DataFrame(X_test)
test_df.to_csv("test.csv")




