import streamlit as st

st.set_page_config(
    page_title="Bank Marketing Campaign App",
)

st.sidebar.success("Select a page above.")

st.title('💰Bank Marketing Campaign📈')

st.header('About')

st.write("""
This app allows users to train and evaluate models by simply uploading their CSV data file.
""")

st.header('Available Apps')
st.write("""
    - **Data Analysis**: Explore and analyze the data.
    - **Models Training and Evaluation**: Model training and evaluate performance.
    - **Prediction**: Predict whether someone would end up subscribing to the term deposit.
""")
