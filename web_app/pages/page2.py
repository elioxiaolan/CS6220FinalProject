import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df['y'] = df['y'].eq('yes').mul(1)
    df['y_str'] = df['y'].astype(str)
    df['age'] = pd.to_numeric(df['age'], errors='coerce')  # Ensure age is numeric
    df['age_range'] = pd.qcut(df['age'], 10, duplicates='drop')
    return df

# User input sidebar
st.sidebar.header('User Input Features')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    train = load_data(uploaded_file)
    analysis_type = st.sidebar.selectbox('Select Analysis Type', (
        'Response Distribution', 
        'Education Analysis', 
        'Job Analysis', 
        'Credit Default Analysis', 
        'Housing Analysis', 
        'Loan Analysis', 
        'Marital Status Analysis', 
        'Age Analysis'
    ))

    # Data analysis dashbroad
    st.title('Data Analysis Dashboard')

    if analysis_type == 'Response Distribution':
        st.header('Response Variable Distribution')
        fig, ax = plt.subplots()
        sns.countplot(x='y', data=train, palette='viridis')
        st.pyplot(fig)

    elif analysis_type == 'Education Analysis':
        st.header('Analysis by Education Level')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y='education', hue='y_str', data=train)
        st.pyplot(fig)

    elif analysis_type == 'Job Analysis':
        st.header('Analysis by Job Role')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y='job', hue='y_str', data=train)
        st.pyplot(fig)

    elif analysis_type == 'Credit Default Analysis':
        st.header('Analysis by Credit Default')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y='default', hue='y_str', data=train)
        st.pyplot(fig)

    elif analysis_type == 'Housing Analysis':
        st.header('Analysis by Housing')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y='housing', hue='y_str', data=train)
        st.pyplot(fig)

    elif analysis_type == 'Loan Analysis':
        st.header('Analysis by Loan')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y='loan', hue='y_str', data=train)
        st.pyplot(fig)

    elif analysis_type == 'Marital Status Analysis':
        st.header('Analysis by Marital Status')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y='marital', hue='y_str', data=train)
        st.pyplot(fig)

    elif analysis_type == 'Age Analysis':
        st.header('Analysis by Age')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=train, x='age', hue='y_str', multiple="stack", bins=20, palette='viridis')
        st.pyplot(fig)

else:
    st.info('Waiting for CSV file to be uploaded.')

