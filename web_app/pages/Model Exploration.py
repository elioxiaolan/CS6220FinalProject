import streamlit as st

st.title("Model Exploration")

model_options = [
    'Logistic Regression',
    'KNN',
    'Naive Bayes',
    'Decision Tree',
    'Random Forest',
    'SVM',
    'Linear Discriminant Analysis'
]

analysis_type = st.selectbox('Select Model', model_options)