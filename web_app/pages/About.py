import streamlit as st

st.title("About")

st.header("About the Dataset")

st.write(
    """    
    The data under study here is called **Bank Marketing Dataset (BMD)** and it was found in the **UCI Machine Learning 
    Repository**. The size of the dataset is considerably large, especially if we consider its origin. 
    Data from clients of financial institutions are usually difficult to find, and when found, 
    are rarely available in this quantity. In the BMD data we have 41188 observations, with eighteen features.
    
    This dataset is related with direct marketing campaigns based on phone calls of a Portuguese banking institution. 
    Often, more than one contact to the same client was required, in order to analyze if the product (bank term deposit) 
    would be (yes) or not (no) subscribed.
    """
)

