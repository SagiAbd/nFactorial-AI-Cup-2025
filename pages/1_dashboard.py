"""
Finance Agent App - Dashboard Page
"""
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Dashboard - Finance Agent",
    page_icon="📊",
    layout="wide"
)

st.title("Dashboard 📊")
st.subheader("Financial Overview")

# Placeholder for dashboard content
st.info("This dashboard will show your financial overview and key metrics.")

# Mock data for demonstration
st.markdown("### Monthly Spending")
st.bar_chart({"Food": 400, "Transport": 200, "Entertainment": 150, "Utilities": 300})

st.markdown("### Recent Transactions")
# This would load from data/user_data.csv in a real implementation
transactions = pd.DataFrame({
    "Date": ["2023-11-01", "2023-11-03", "2023-11-05"],
    "Description": ["Grocery Store", "Electricity Bill", "Restaurant"],
    "Amount": [-120.45, -85.20, -65.30],
    "Category": ["Groceries", "Utilities", "Dining"]
})
st.dataframe(transactions)
