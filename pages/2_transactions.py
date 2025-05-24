"""
Finance Agent App - Transactions Page
"""
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Transactions - Finance Agent",
    page_icon="💳",
    layout="wide"
)

st.title("Transactions 💳")
st.subheader("View and Manage Your Transactions")

# File uploader for transaction data
st.markdown("### Upload Transaction Data")
uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    # In a real implementation, this would parse and save the file
else:
    # Show sample data
    st.markdown("### Sample Transactions")
    # This would load from data/user_data.csv in a real implementation
    transactions = pd.DataFrame({
        "Date": ["2023-11-01", "2023-11-03", "2023-11-05", "2023-11-07", "2023-11-10"],
        "Description": ["Grocery Store", "Electricity Bill", "Restaurant", "Gas Station", "Online Shopping"],
        "Amount": [-120.45, -85.20, -65.30, -45.67, -95.99],
        "Category": ["Groceries", "Utilities", "Dining", "Transportation", "Shopping"]
    })
    
    # Display transactions with filters
    st.dataframe(transactions)
    
    # Category filter example
    categories = transactions["Category"].unique().tolist()
    selected_category = st.selectbox("Filter by category", ["All"] + categories)
    
    st.info("This page will allow you to upload, view, and categorize your transactions.")
