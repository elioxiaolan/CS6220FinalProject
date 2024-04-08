import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import logging
from joblib import dump


import sklearn
print(sklearn.__version__)

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess(data):
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

    return [X, y]


def train_and_save(processed_data, model, model_name):
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                            'day_of_week', 'poutcome']
    numerical_features = ['age', 'duration', 'campaign']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False), categorical_features)
        ])

    X_train, X_test, y_train, y_test = train_test_split(processed_data[0], processed_data[1], test_size=0.3,
                                                        random_state=42)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

    pipeline.fit(X_train, y_train)

    file_name = f"../models/{model_name}.joblib"
    dump(pipeline, file_name)


def main():
    data = pd.read_csv("../new_train2.csv")
    processed_data = preprocess(data)

    model_options = ["Logistic Regression", "Random Forest", "SVM"]

    for model_option in model_options:
        if model_option == "Logistic Regression":
            model = LogisticRegression(solver='liblinear', max_iter=1000)
            model_name = "Logistic Regression"
        elif model_option == "Random Forest":
            model = RandomForestClassifier()
            model_name = "Random Forest"
        elif model_option == "SVM":
            model = SVC(probability=True)
            model_name = "SVM"

        train_and_save(processed_data, model, model_name)

if __name__ == "__main__":
    main()
