import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, \
    ConfusionMatrixDisplay, precision_recall_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_and_train(data, model, model_name):
    # Checking missing values
    if data.isnull().any().any():
        st.warning("Missing values detected. Auto-imputing missing values.")
        for column in data.columns:
            # Categorical
            if data[column].dtype == 'object':
                data[column].fillna(data[column].mode()[0], inplace=True)
            # Numerical
            else:
                data[column].fillna(data[column].median(), inplace=True)

    data['label'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0)
    data.drop('y', axis=1, inplace=True)

    X = data.drop('label', axis=1)
    y = data['label']

    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                            'day_of_week', 'poutcome']
    numerical_features = ['age', 'duration', 'campaign']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False), categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    try:
        pipeline.fit(X_train, y_train)
        cv_score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        st.write(f"Cross-Validation Accuracy: {np.mean(cv_score):.4f} ± {np.std(cv_score):.4f}")
    except Exception as e:
        logger.error("Error during model training or cross-validation: %s", e)
        st.error(f"Error occurred during model training or cross-validation. Error: {e}")

    # Model evaluation
    try:
        y_pred = pipeline.predict(X_test)
        st.write(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        st.table(pd.DataFrame(report_dict).transpose())

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots()
        disp.plot(ax=ax)
        st.pyplot(fig)

        if hasattr(model, "predict_proba"):
            probas_ = pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, probas_)
            roc_auc = auc(fpr, tpr)

            # ROC curve
            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})', lw=2)
            plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {model_name}')
            plt.legend(loc="lower right")
            st.pyplot(plt)

            # Precision recall curve
            precision, recall, _ = precision_recall_curve(y_test, probas_)

            plt.figure()
            plt.plot(recall, precision, marker='.', label=model_name)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve for {model_name}')
            plt.legend()
            st.pyplot(plt)

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        st.error("An error occurred during model evaluation. Error:", {e})
        # st.error(f"An error occurred during model evaluation. Error: {e}")


# Streamlit UI
st.title('Model Training and Performance Evaluation Dashboard')

st.header('About This App')
st.write("""
This app allows users to train and evaluate models by simply uploading their CSV data file.
""")

st.header('Available Models')
st.write("""
    - **Logistic Regression**
    - **Random Forest**
    - **SVM**
""")

st.write("""
Might required additional time to generate result, especially for SVM.
""")

st.header('Getting Start')
st.write("""
1. Upload your CSV file.
2. Select the type of model you want to perform.
3. View the results on the dashboard.
""")

uploaded_file = st.file_uploader("Upload your input CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    model_option = st.selectbox("Select a model", ("Logistic Regression", "Random Forest", "SVM"))

    if model_option == "Logistic Regression":
        model = LogisticRegression(solver='liblinear', max_iter=1000)
        model_name = "Logistic Regression"
    elif model_option == "Random Forest":
        model = RandomForestClassifier()
        model_name = "Random Forest"
    elif model_option == "SVM":
        model = SVC(probability=True)
        model_name = "SVM"

    if st.button("Train and Evaluate Model"):
        preprocess_and_train(data, model, model_name)
