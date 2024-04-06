import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Load and preprocess data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['y'] = df['y'].eq('yes').mul(1)
    df['y_str'] = df['y'].astype(str)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')  # Ensure age is numeric
    df['age_range'] = pd.qcut(df['age'], 10, duplicates='drop')
    return df


# Encode categorical variables for correlation
def encode_features(df):
    le = LabelEncoder()
    features_to_encode = ['education', 'job', 'default', 'housing', 'loan', 'marital']
    encoded_features = {}
    for feature in features_to_encode:
        le.fit(df[feature].unique())
        encoded_feature = le.transform(df[feature]).tolist()
        encoded_features[feature] = encoded_feature
    encoded_features['age'] = df['age'].tolist()
    encoded_features['duration'] = df['duration'].tolist()
    encoded_features['y'] = df['y'].tolist()
    return pd.DataFrame(encoded_features)


# Plotting functions
def plot_response_distribution(train):
    fig, ax = plt.subplots()
    sns.countplot(x='y', data=train, palette='viridis')
    st.pyplot(fig)


def plot_education_analysis(train):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='education', hue='y_str', data=train)
    st.pyplot(fig)


def plot_job_analysis(train):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='job', hue='y_str', data=train)
    st.pyplot(fig)


def plot_default_analysis(train):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='default', hue='y_str', data=train)
    st.pyplot(fig)


def plot_housing_analysis(train):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='housing', hue='y_str', data=train)
    st.pyplot(fig)


def plot_loan_analysis(train):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='loan', hue='y_str', data=train)
    st.pyplot(fig)


def plot_marital_status_analysis(train):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(y='marital', hue='y_str', data=train)
    st.pyplot(fig)


def plot_age_analysis(train):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=train, x='age', hue='y_str', multiple="stack", bins=20, palette='viridis')
    st.pyplot(fig)


def show_correlation_heatmap(df):
    plt.figure(figsize=(10, 9))
    sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', linewidths=0.2, annot_kws={'size':13})
    st.pyplot(plt.gcf())


def show_age_distribution(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df['age'])
    plt.title('Age Distribution')
    st.pyplot(plt.gcf())


def show_job_distribution(df):
    plt.figure(figsize=(20, 6))
    sns.histplot(df['job'], bins=30, kde=False)
    plt.title('Job Distribution')
    st.pyplot(plt.gcf())


def show_correlation_matrix(df):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f")
    st.pyplot(plt.gcf())


# App UI
st.sidebar.header('User Input')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

analysis_options = [
    'Home',  
    'Response Distribution', 'Education Analysis', 'Job Analysis',
    'Credit Default Analysis', 'Housing Analysis', 'Loan Analysis',
    'Marital Status Analysis', 'Age Analysis', 'Correlation Heatmap',
    'Age Distribution', 'Job Distribution', 'Numerical Feature Correlations'
]

analysis_type = st.sidebar.selectbox('Select Analysis Type', analysis_options)

if analysis_type == 'Home':
    st.title('Welcome to the Data Analysis Dashboard')

    st.header('About This App')
    st.write("""
    This data analysis app allows users to upload their data and perform analyses by simply uploading the CSV data file.
    """)

    st.header('Available Analyses')
    st.write("""
    - **Response Distribution**
    - **Education Analysis**
    - **Job Analysis**
    - **Credit Default Analysis**
    - **Housing Analysis**
    - **Loan Analysis**
    - **Marital Status Analysis**
    - **Age Analysis**
    - **Correlation Heatmap**
    - **Age Distribution**
    - **Job Distribution**
    - **Numerical Feature Correlations** 
    """)

    st.header('Getting Start')
    st.write("""
    1. Use the sidebar to upload your CSV file.
    2. Select the type of analysis you want to perform.
    3. View the results on the dashboard.
    """)

elif uploaded_file is not None:
    train = load_data(uploaded_file)
    if analysis_type != 'Home':
        encoded_df = encode_features(train)  
        
    st.title('Data Analysis Dashboard')

    if analysis_type == 'Response Distribution':
        st.header('Response Variable Distribution')
        plot_response_distribution(train)
   
    elif analysis_type == 'Job Analysis':
        st.header('Analysis by Job Role')
        plot_job_analysis(train)

    elif analysis_type == 'Education Analysis':
        st.header('Analysis by Education Level')
        plot_education_analysis(train)

    elif analysis_type == 'Credit Default Analysis':
        st.header('Analysis by Credit Default')
        plot_default_analysis(train)

    elif analysis_type == 'Housing Analysis':
        st.header('Analysis by Housing')
        plot_housing_analysis(train)

    elif analysis_type == 'Loan Analysis':
        st.header('Analysis by Loan')
        plot_loan_analysis(train)

    elif analysis_type == 'Marital Status Analysis':
        st.header('Analysis by Marital Status')
        plot_marital_status_analysis(train)

    elif analysis_type == 'Age Analysis':
        st.header('Analysis by Age')
        plot_age_analysis(train)

    elif analysis_type == 'Correlation Heatmap':
        st.header('Correlation Heatmap')
        show_correlation_heatmap(encoded_df)

    elif analysis_type == 'Age Distribution':
        st.header('Age Distribution')
        show_age_distribution(train)

    elif analysis_type == 'Job Distribution':
        st.header('Job Distribution')
        show_job_distribution(train)

    elif analysis_type == 'Numerical Feature Correlations':
        st.header('Correlation Matrix for Numerical Features')
        show_correlation_matrix(train)



else:
    if analysis_type != 'Home':
        st.info('Waiting for CSV file to be uploaded.')
