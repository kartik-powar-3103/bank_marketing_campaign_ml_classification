
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="ML Classification Models",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Bank Marketing Classification - ML Models Comparison</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Comparison of Model", "Predict", "Dataset Info"]
)

# Load models and metrics
@st.cache_data
def load_models():
    """Load all trained models"""
    models = {}
    model_files = {
        'Logistic Regression': 'model/logistic_regression.pkl',
        'Decision Tree': 'model/decision_tree.pkl',
        'KNN': 'model/knn.pkl',
        'Naive Bayes': 'model/naive_bayes.pkl',
        'Random Forest': 'model/random_forest.pkl',
        'XGBoost': 'model/xgboost.pkl'
    }
    
    
    for model_name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                models[model_name] = joblib.load(filepath)
            except Exception as e:
                st.warning(f"Could not load {model_name}: {str(e)}")
    
    return models

@st.cache_data
def load_metrics():
    """Load saved metrics"""
    metrics_path = 'model_comparison.csv'
    if os.path.exists(metrics_path):
        try:
            metrics_df = pd.read_csv(metrics_path, index_col=0)
            return metrics_df
        except Exception as e:
            st.warning(f"Could not load metrics: {str(e)}")
            return None
    return None

# Load data
try:
    models = load_models()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    models = {}, None

try:
    metrics_df = load_metrics()
except Exception as e:
    st.error(f"Error loading metrics: {str(e)}")
    metrics_df = None

# Page 1: Comparison of Model
if page == "Comparison of Model":
    st.header("Model Performance Comparison")

    if (metrics_df is None or metrics_df.empty) and not models:
        st.error("**Models and metrics not found!**")

    elif metrics_df is not None and not metrics_df.empty:

        # =======================
        # Comparison Table
        # =======================
        st.subheader("Model Evaluation Metrics")

        comparison_df = metrics_df.reset_index().rename(columns={"index": "ML Model Name"})
        display_cols = ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
        comparison_df[display_cols] = comparison_df[display_cols].round(4)
        comparison_df.index = range(1, len(comparison_df) + 1)

        st.dataframe(comparison_df[display_cols], use_container_width=True)

        # =======================
        # Model Selection
        # =======================
        st.subheader("Select Model for Detailed Analysis")
        selected_model = st.selectbox("Choose a model", metrics_df.index.tolist())
        selected_metrics = metrics_df.loc[selected_model]

        col1, col2 = st.columns(2)

        # =======================
        # Metrics Panel
        # =======================
        with col1:
            st.markdown("## Model Metrics")

            baseline = 50.0
            metrics_list = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score"]
            mcol1, mcol2 = st.columns(2)

            for i, metric in enumerate(metrics_list):
                val = selected_metrics[metric] * 100
                delta = val - baseline
                target = mcol1 if i % 2 == 0 else mcol2
                target.metric(metric, f"{val:.2f}%", f"{delta:+.2f}%")

            st.divider()
            st.metric("MCC", f"{selected_metrics['MCC']:.3f}")

        # =======================
        # Confusion Matrix
        # =======================
        with col2:
            st.markdown("## Confusion Matrix")

            tp, fp, fn, tn = map(int, (
                selected_metrics["TP"],
                selected_metrics["FP"],
                selected_metrics["FN"],
                selected_metrics["TN"]
            ))

            # KPI tiles
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("TP", tp)
            k2.metric("FP", fp)
            k3.metric("FN", fn)
            k4.metric("TN", tn)

            st.divider()

            show_percent = st.toggle("Show percentages instead of raw counts", value=False)

            cm = np.array([[tn, fp], [fn, tp]])
            cm_display = cm / cm.sum() * 100 if show_percent else cm
            fmt = ".1f" if show_percent else "d"

            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm_display,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                linewidths=2,
                linecolor="white",
                square=True,
                cbar=True,
                xticklabels=["Pred −", "Pred +"],
                yticklabels=["Actual −", "Actual +"],
                ax=ax,
                annot_kws={"size": 16, "weight": "bold"}
            )

            # Green / Red annotations
            for i, text in enumerate(ax.texts):
                r, c = divmod(i, 2)
                if (r == 0 and c == 0) or (r == 1 and c == 1):
                    text.set_color("green")   # TN / TP
                else:
                    text.set_color("red")     # FP / FN

            ax.set_title(f"Confusion Matrix – {selected_model}", fontsize=14, fontweight="bold", pad=14)
            ax.set_xlabel("")
            ax.set_ylabel("")
            st.pyplot(fig, use_container_width=True)

            # Derived rates
            total = tp + tn + fp + fn
            tpr = tp / (tp + fn)
            tnr = tn / (tn + fp)
            fpr = fp / (fp + tn)

            r1, r2 = st.columns(2)
            r1.metric("Recall (TPR)", f"{tpr * 100:.2f}%")
            r2.metric("Specificity (TNR)", f"{tnr * 100:.2f}%")

            r3, r4 = st.columns(2)
            r3.metric("False Positive Rate", f"{fpr * 100:.2f}%")
            r4.metric("Accuracy (CM)", f"{(tp + tn) / total * 100:.2f}%")

        # =======================
        # Side-by-Side Comparison
        # =======================
        st.subheader("Compare Two Models")

        c1, c2 = st.columns(2)
        model_a = c1.selectbox("Model A", metrics_df.index, key="model_a")
        model_b = c2.selectbox("Model B", metrics_df.index, key="model_b")

        compare_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]
        compare_df = pd.DataFrame({
            model_a: metrics_df.loc[model_a, compare_cols],
            model_b: metrics_df.loc[model_b, compare_cols],
        })

        compare_df = compare_df.applymap(lambda x: f"{x * 100:.2f}%" if x <= 1 else f"{x:.3f}")
        st.dataframe(compare_df, use_container_width=True)

    else:
        st.warning("No metrics found. Please train the models first.")

# Page 2: Predict
elif page == "Predict":
    st.header("Upload Test Data and Make Predictions")
    
    st.info("Upload a CSV file with test data. The file should have the same features as the training data (excluding the target column).")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload test data CSV file"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded data
            try:
                test_data = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(test_data)} rows, {len(test_data.columns)} columns")
                
                # Display data preview
                st.subheader("Data Preview")
                st.dataframe(test_data.head(10), width='stretch')
                
                # Show data info
                with st.expander("Data Information"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Shape:**", test_data.shape)
                        st.write("**Columns:**", list(test_data.columns))
                    with col2:
                        st.write("**Data Types:**")
                        st.write(test_data.dtypes)
                        if test_data.isnull().any().any():
                            st.warning("Missing values detected")
                            st.write("Missing values per column:")
                            st.write(test_data.isnull().sum())
            except Exception as e:
                st.error(f"Error reading CSV file: {str(e)}")
                st.info("Please ensure the file is a valid CSV format.")
                st.stop()
            
            # Model selection
            if models:
                st.subheader("Select Model for Prediction")
                selected_model = st.selectbox(
                    "Choose a model",
                    list(models.keys())
                )
                
                if st.button("Make Predictions"):
                    try:
                        model = models[selected_model]
                        
                        # Validate data columns
                        expected_columns = ['age', 'job', 'marital', 'education', 'default', 'balance',
                                            'housing', 'loan', 'contact', 'day', 'month', 'duration',
                                            'campaign', 'pdays', 'previous','poutcome']
                        
                        missing_cols = set(expected_columns) - set(test_data.columns)
                        if missing_cols:
                            st.error(f"Missing columns: {', '.join(missing_cols)}")
                            st.info(f"Required columns: {', '.join(expected_columns)}")
                            st.stop()
                        
                        # Check for extra columns (warn but continue)
                        extra_cols = set(test_data.columns) - set(expected_columns)
                        if extra_cols:
                            st.warning(f"Extra columns detected (will be ignored): {', '.join(extra_cols)}")
                        
                        # Select only required columns in correct order
                        test_data_clean = test_data[expected_columns].copy()
                        
                        # Check for missing values
                        if test_data_clean.isnull().any().any():
                            st.warning("Missing values detected. Filling with column means.")
                            test_data_clean = test_data_clean.fillna(test_data_clean.mean())
                        
                        X_test = test_data_clean.copy()
                        
                        # Make predictions
                        with st.spinner('Making predictions...'):
                            predictions = model.predict(X_test)
                            prediction_proba = model.predict_proba(X_test)
                        
                        # Load label encoder if exists to convert predictions back to original labels
                        reverse_map = {1: "yes", 0: "no"}
                        predictions_original = pd.Series(predictions).map(reverse_map)
                        
                        # Display results
                        st.subheader("Predictions")
                        st.success(f"Successfully predicted {len(predictions)} samples")
                        
                        results_df = pd.DataFrame({
                            'Sample': range(1, len(predictions) + 1),
                            'Predicted Quality': predictions_original,
                            'Confidence': np.max(prediction_proba, axis=1)
                        })
                        
                        # Format confidence as percentage
                        results_df['Confidence'] = results_df['Confidence'].apply(lambda x: f"{x*100:.2f}%")
                        
                        st.dataframe(results_df, width='stretch', hide_index=True)
                        
                        # Show prediction statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Predictions", len(predictions))
                        with col2:
                            st.metric("Most Common Prediction", pd.Series(predictions_original).mode()[0])
                        with col3:
                            avg_confidence = np.max(prediction_proba, axis=1).mean()
                            st.metric("Average Confidence", f"{avg_confidence*100:.2f}%")
                        
                        # Download predictions
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name='predictions.csv',
                            mime='text/csv'
                        )
                        
                    except Exception as e:
                        st.error(f"Error making predictions: {str(e)}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())
            else:
                st.warning("No models found. Please train the models first.")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        # Show example format
        st.subheader("Expected Data Format")
        st.info(
        """
        **The CSV file should contain the following columns:**

        - age  
        - job  
        - marital  
        - education  
        - default  
        - balance  
        - housing  
        - loan  
        - contact  
        - day  
        - month  
        - duration  
        - campaign  
        - pdays  
        - previous  
        - poutcome  
        """
        )


# Page 3: Dataset Info
elif page == "Dataset Info":
    st.header("Dataset Information")
    
    # Dataset Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Bank Marketing Dataset
        
        **Data Source** Kaggle (https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data)
        
        **Description:**
        This is the classic marketing bank dataset uploaded originally in the UCI Machine Learning 
        Repository. The dataset gives you information about a marketing campaign of a financial 
        institution in which you will have to analyze in order to find ways to look for future 
        strategies in order to improve future marketing campaigns for the bank.
        """)
    
    with col2:
        st.info("""
        **Quick Stats:**
        - Instances: 11162
        - Features: 16
        - Classes: 2
        - Type: Classification
        """)
    
    # Feature Information
    st.subheader("Feature Information")
    
    features_df = pd.DataFrame({
        'Feature': [
            'age',
            'job',
            'marital',
            'education',
            'default',
            'balance',
            'housing',
            'loan',
            'contact',
            'day',
            'month',
            'duration',
            'campaign',
            'pdays',
            'previous',
            'poutcome'
        ]
    })
    
    st.dataframe(features_df, width='stretch', hide_index=True)
    
    # Dataset Exploration
    st.subheader("Dataset Exploration")
    
    # Try to load the actual dataset
    dataset_loaded = False
    wine_data = None
    
    # Try multiple possible locations
    possible_paths = [
        'train.csv',
        'test.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                bank_data = pd.read_csv(path, index_col=0)
                bank_data["deposit"] = bank_data["deposit"].map({"yes": 1, "no": 0})
                dataset_loaded = True
                st.success(f"Loaded dataset from: {path}")
                break
            except:
                try:
                    # Fallback to comma delimiter
                    bank_data = pd.read_csv(path)
                    dataset_loaded = True
                    st.success(f"Loaded dataset from: {path}")
                    break
                except:
                    continue
    
    if dataset_loaded and bank_data is not None:
        # Dataset shape
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", bank_data.shape[0])
        with col2:
            st.metric("Total Columns", bank_data.shape[1])
        with col3:
            st.metric("Memory Usage", f"{bank_data.memory_usage(deep=True).sum() / 1024:.2f} KB")
        with col4:
            st.metric("Missing Values", bank_data.isnull().sum().sum())
        
        # Data Preview Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "First Rows", 
            "Last Rows", 
            "Statistics", 
            "Distribution",
            "Correlations"
        ])
        
        with tab1:
            st.write("**First 10 rows of the dataset:**")
            st.dataframe(bank_data.head(10), width='stretch')
        
        with tab2:
            st.write("**Last 10 rows of the dataset:**")
            st.dataframe(bank_data.tail(10), width='stretch')
        
        with tab3:
            st.write("**Statistical Summary:**")
            st.dataframe(bank_data.describe(), width='stretch')
            
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': bank_data.columns,
                'Data Type': bank_data.dtypes.values,
                'Non-Null Count': bank_data.count().values,
                'Null Count': bank_data.isnull().sum().values
            })
            st.dataframe(dtype_df, width='stretch', hide_index=True)
        
        with tab4:
            st.write("**Target Variable Distribution:**")
            if 'deposit' in bank_data.columns:
                deposit_counts = bank_data['deposit'].value_counts().sort_index()
                
                fig, ax = plt.subplots(figsize=(10, 5))
                deposit_counts.plot(kind='bar', ax=ax, color='steelblue')
                ax.set_xlabel('Campaign Response')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Campaign Response')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Bank Deposit Campaign Response Distribution:**")
                    st.dataframe(
                        pd.DataFrame({
                            'Deposits': deposit_counts.index,
                            'Count': deposit_counts.values,
                            'Percentage': (deposit_counts.values / len(bank_data) * 100).round(2)
                        }),
                        width='stretch',
                        hide_index=True
                    )
                with col2:
                    st.metric("Most Common Campaign Response", deposit_counts.idxmax())
                    st.metric("Average Campaign Response", f"{bank_data['deposit'].mean():.2f}")
                    st.metric("Campaign Response Std Dev", f"{bank_data['deposit'].std():.2f}")
        
        with tab5:
            st.write("**Feature Correlation Matrix:**")
            
            # Calculate correlation matrix
            corr_matrix = bank_data.corr()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title('Feature Correlation Heatmap')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            if 'deposit' in bank_data.columns:
                st.write("**Correlation with Target:**")
                deposit_corr = corr_matrix['deposit'].drop('deposit').sort_values(ascending=False)
                corr_df = pd.DataFrame({
                    'Feature': deposit_corr.index,
                    'Correlation': deposit_corr.values
                })
                st.dataframe(corr_df, width='stretch', hide_index=True)
    else:
        st.warning("Dataset file not found. Please ensure 'bank.csv' is in the project directory.")
        st.info("""
        You can download the dataset from:
        - [Kaggle - Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data)
        """)
    
    # Models Information
    st.subheader("Implemented Models")
    
    models_info = pd.DataFrame({
        'Model': [
            'Logistic Regression',
            'Decision Tree',
            'K-Nearest Neighbors (KNN)',
            'Naive Bayes (Gaussian)',
            'Random Forest',
            'XGBoost'
        ],
        'Type': [
            'Linear',
            'Tree-based',
            'Instance-based',
            'Probabilistic',
            'Ensemble',
            'Ensemble (Boosting)'
        ],
        'Key Characteristics': [
            'Fast, interpretable, works well with linear relationships',
            'Non-linear, interpretable, prone to overfitting',
            'Non-parametric, sensitive to feature scaling',
            'Fast, works well with independent features',
            'Reduces overfitting, handles non-linear relationships',
            'High performance, handles complex patterns'
        ]
    })
    
    st.dataframe(models_info, width='stretch', hide_index=True)
    
    # Evaluation Metrics
    st.subheader("📏 Evaluation Metrics")
    
    metrics_info = pd.DataFrame({
        'Metric': [
            'Accuracy',
            'AUC Score',
            'Precision',
            'Recall',
            'F1 Score',
            'MCC'
        ],
        'Description': [
            'Overall correctness of predictions',
            'Area under ROC curve - model discrimination ability',
            'Accuracy of positive predictions',
            'Coverage of actual positive cases',
            'Harmonic mean of precision and recall',
            'Matthews Correlation Coefficient - balanced measure'
        ],
        'Range': [
            '0 to 1 (higher is better)',
            '0 to 1 (higher is better)',
            '0 to 1 (higher is better)',
            '0 to 1 (higher is better)',
            '0 to 1 (higher is better)',
            '-1 to 1 (higher is better)'
        ]
    })
    
    st.dataframe(metrics_info, width='stretch', hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>ML Assignment 2 - Classification Models Comparison</div>",
    unsafe_allow_html=True
)