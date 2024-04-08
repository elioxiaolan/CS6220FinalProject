import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
from joblib import load

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_age_input(age_input):
    try:
        age = int(age_input)
        if age < 0:
            st.warning("Age cannot be negative. Please enter a valid age.")
            age = 27  # Greatest number of people among all ages (mode)
    except ValueError:
        if age_input:
            st.warning("Please enter a valid number for age.")
        age = 27

    return age


def process_job(job_input):
    job_map = {
        "Administrator": "admin.",
        "Blue Collar": "blue-collar",
        "Entrepreneur": "entrepreneur",
        "Housemaid": "housemaid",
        "Management": "management",
        "Retired": "retired",
        "Self Employed": "self-employed",
        "Services": "services",
        "Student": "student",
        "Technician": "technician",
        "Unemployed": "unemployed",
        "Unknown": "unknown"
    }

    if job_input not in job_map:
        return "unknown"
    else:
        return job_map[job_input]


def process_marital_status(marital_status_input):
    marital_map = {
        "Divorced": "divorced",
        "Married": "married",
        "Single": "single",
        "Unknown": "unknown"
    }

    if marital_status_input not in marital_map:
        return "unknown"
    else:
        return marital_map[marital_status_input]


def process_education_status(education_status_input):
    education_map = {
        "4 Years of Basic Education (Quatro Anos de Ensino Básico)": "basic.4y",
        "6 Years of Basic Education (Seis Anos de Ensino Básico)": "basic.6y",
        "9 Years of Basic Education (Nove Anos de Ensino Básico)": "basic.9y",
        "Secondary Education (Ensino Secundário)": "high.school",
        "Illiterate (Analfabeto)": "illiterate",
        "Professional Course (Curso Profissional)": "professional.course",
        "University Degree (Grau Universitário)": "university.degree",
        "Unknown": "unknown"
    }

    if education_status_input not in education_map:
        return "unknown"
    else:
        return education_map[education_status_input]


def process_default_status(default_status_input):
    default_map = {
        "Yes": "yes",
        "No": "no",
        "Unknown": "unknown"
    }

    if default_status_input not in default_map:
        return "unknown"
    else:
        return default_map[default_status_input]


def process_housing_status(housing_status_input):
    housing_map = {
        "Yes": "yes",
        "No": "no",
        "Unknown": "unknown"
    }

    if housing_status_input not in housing_map:
        return "unknown"
    else:
        return housing_map[housing_status_input]


def process_loan_status(loan_status_input):
    loan_map = {
        "Yes": "yes",
        "No": "no",
        "Unknown": "unknown"
    }

    if loan_status_input not in loan_map:
        return "unknown"
    else:
        return loan_map[loan_status_input]


def process_contact(contact_type_input):
    contact_type_map = {
        "Cellular": "cellular",
        "Telephone": "telephone",
    }

    if contact_type_input not in contact_type_map:
        return "unknown"
    else:
        return contact_type_map[contact_type_input]


def process_month(month_input):
    month_map = {
        "January": "jan",
        "February": "feb",
        "March": "mar",
        "April": "apr",
        "May": "may",
        "June": "jun",
        "July": "jul",
        "August": "aug",
        "September": "sep",
        "October": "oct",
        "November": "nov",
        "December": "dec"
    }

    if month_input not in month_map:
        return "unknown"
    else:
        return month_map[month_input]


def process_day_of_week(day_of_week_input):
    weekday_map = {
        "Monday": "mon",
        "Tuesday": "tue",
        "Wednesday": "web",
        "Thursday": "thu",
        "Friday": "fri",
    }

    if day_of_week_input not in weekday_map:
        return "unknown"
    else:
        return weekday_map[day_of_week_input]


def process_duration_input(duration_input):
    try:
        duration = int(duration_input)
        if duration < 0:
            st.warning("Duration cannot be negative. Please enter a valid duration.")
            duration = 0
    except ValueError:
        if duration_input:
            st.warning("Please enter a valid number for duration.")
        duration = 27

    return duration


def process_number_of_contact(number_of_contact_input):
    try:
        number_of_contact = int(number_of_contact_input)
        if number_of_contact < 0:
            st.warning("Number of Contact cannot be negative. Please enter a valid number.")
            number_of_contact = 0
    except ValueError:
        if number_of_contact_input:
            st.warning("Please enter a valid number for Number of Contact.")
        number_of_contact = 0

    return number_of_contact


def process_previous_outcome(previous_outcome_input):
    outcome_map = {
        "Failure": "failure",
        "Non-Existent": "nonexistent",
        "Success": "success",
    }

    if previous_outcome_input not in outcome_map:
        return "unknown"
    else:
        return outcome_map[previous_outcome_input]


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

    return pipeline


def predict_single(input_df):
    if input_model in input_models:
        model_path = f"../models/{input_model}.joblib"
        model = load(model_path)

        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[:, 1]

        predicted_label = 'Subscribe' if prediction[0] == 1 else 'Not Subscribe'
        st.write(f"Predicted Label: {predicted_label}, Probability: {probability[0]:.4f}")
    else:
        st.warning("Please select a valid model to predict.")


def predict_multiple(input_df):
    if input_model in input_models:
        model_path = f"../models/{input_model}.joblib"
        model = load(model_path)

        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]

        input_df['predicted_label'] = predictions
        input_df['predicted_label'] = input_df['predicted_label'].apply(lambda x: 'Subscribe' if x == 1 else 'Not Subscribe')
        input_df['probability'] = probabilities

        st.write("Prediction Results:")
        st.dataframe(input_df)
    else:
        st.warning("Please select a valid model to predict.")


# Streamlit UI
st.title("Model Prediction")

st.header("About This App")
st.write("""This app provides users prediction feature so that they can both manually choose various options and 
upload their CSV data files based on their respective circumstances.""")

st.header("Available Models")
st.write("""
    - **Logistic Regression**
    - **Random Forest**
    - **SVM**
""")

input_models = [
    "  ",
    "Logistic Regression",
    "Random Forest",
    "SVM"
]

input_model = st.selectbox("Please select a model: ", input_models)

input_types = [
    "  ",
    "Manually Input",
    "Upload CSV Data Files"
]

input_type = st.selectbox("Please select an option: ", input_types)

if input_type == "Manually Input":
    st.header("Enter Your Information: ")

    # Age
    age_input = st.text_input("Age")
    age = process_age_input(age_input)

    # Job
    jobs = ["  ", "Administrator", "Blue Collar", "Entrepreneur", "Housemaid", "Management", "Retired", "Self Employed",
            "Services", "Student", "Technician", "Unemployed", "Unknown"]
    job_input = st.selectbox(
        "Job",
        jobs
    )
    job = process_job(job_input)

    # Marital
    marital_statuses = ["", "Divorced", "Married", "Single", "Unknown"]
    marital_status_input = st.selectbox("Marital Status", marital_statuses)
    marital_status = process_marital_status(marital_status_input)

    # Education
    education_statuses = ["  ", "4 Years of Basic Education (Quatro Anos de Ensino Básico)",
                          "6 Years of Basic Education (Seis Anos de Ensino Básico)",
                          "9 Years of Basic Education (Nove Anos de Ensino Básico)",
                          "Secondary Education (Ensino Secundário)", "Illiterate (Analfabeto)",
                          "Professional Course (Curso Profissional)", "University Degree (Grau Universitário)",
                          "Unknown"]
    education_status_input = st.selectbox("Education", education_statuses)
    education_status = process_education_status(education_status_input)

    # Default
    default_status_input = st.selectbox("Has Credit in Default?", ["  ", "Yes", "No", "Unknown"])
    default_status = process_default_status(default_status_input)

    # Housing
    housing_status_input = st.selectbox("Has Housing Loan?", ["  ", "Yes", "No", "Unknown"])
    housing_status = process_housing_status(housing_status_input)

    # Loan
    loan_status_input = st.selectbox("Has Personal Loan?", ["  ", "Yes", "No", "Unknown"])
    loan_status = process_loan_status(loan_status_input)

    # Contact Type
    contact_type_input = st.selectbox("Contact Communication Type", ["  ", "Cellular", "Telephone"])
    contact_type = process_contact(contact_type_input)

    # Month
    month_input = st.selectbox("Last Contact Month",
                               ["  ", "January", "February", "March", "April", "May", "June", "July", "August",
                                "September",
                                "October", "November", "December"])
    month = process_month(month_input)

    # Day of the week
    day_of_week_input = st.selectbox("Last Contact Day of the Week", ["  ", "Monday", "Tuesday", "Wednesday",
                                                                      "Thursday", "Friday"])
    day_of_week = process_day_of_week(day_of_week_input)

    # Contact duration
    duration_input = st.text_input("Last Contact Duration (seconds)")
    duration = process_duration_input(duration_input)

    # Number of contacts
    number_of_contact_input = st.text_input("Number of Contacts Performed During this Campaign")
    number_of_contact = process_number_of_contact(number_of_contact_input)

    # Previous Outcome
    previous_outcome_input = st.selectbox("Outcome of the Previous Marketing Campaign", ["  ", "Failure",
                                                                                         "Non-Existent", "Success"])
    previous_outcome = process_previous_outcome(previous_outcome_input)

    input_data = {
        "age": [age],
        "job": [job],
        "marital": [marital_status],
        "education": [education_status],
        "default": [default_status],
        "housing": [housing_status],
        "loan": [loan_status],
        "contact": [contact_type],
        "month": [month],
        "day_of_week": [day_of_week],
        "duration": [duration],
        "campaign": [number_of_contact],
        "poutcome": [previous_outcome]
    }

    input_df = pd.DataFrame.from_dict(input_data)

    if st.button("Predict"):
        predict_single(input_df)
elif input_type == "Upload CSV Data Files":
    st.header("Upload Your CSV Files: ")
    uploaded_file = st.file_uploader("Upload your input CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        processed_data = {
            'age': data['age'].apply(process_age_input),
            'job': data['job'].map(lambda x: process_job(x)),
            'marital': data['marital'].map(lambda x: process_marital_status(x)),
            'education': data['education'].map(lambda x: process_education_status(x)),
            'default': data['default'].map(lambda x: process_default_status(x)),
            'housing': data['housing'].map(lambda x: process_housing_status(x)),
            'loan': data['loan'].map(lambda x: process_loan_status(x)),
            'contact': data['contact'].map(lambda x: process_contact(x)),
            'month': data['month'].map(lambda x: process_month(x)),
            'day_of_week': data['day_of_week'].map(lambda x: process_day_of_week(x)),
            'duration': data['duration'].apply(process_duration_input),
            'campaign': data['campaign'].apply(process_number_of_contact),
            'poutcome': data['poutcome'].map(lambda x: process_previous_outcome(x)),
        }
        input_df = pd.DataFrame(processed_data)

        if st.button("Predict"):
            predict_multiple(input_df)
