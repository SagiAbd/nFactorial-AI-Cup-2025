"""
Finance Agent App - Budget Goals Page
"""
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Budget Goals - Finance Agent",
    page_icon="🎯",
    layout="wide"
)

st.title("Budget Goals 🎯")
st.subheader("Set and Track Your Financial Goals")

# Placeholder for budget goals
st.markdown("### Your Budget Goals")

# Mock data for demonstration
budget_data = pd.DataFrame({
    "Category": ["Groceries", "Dining", "Entertainment", "Transportation", "Shopping"],
    "Monthly Budget": [500, 300, 200, 250, 400],
    "Current Spending": [420, 250, 150, 180, 320],
    "Remaining": [80, 50, 50, 70, 80]
})

st.dataframe(budget_data)

# Budget creation form
st.markdown("### Create New Budget Goal")
with st.form("budget_form"):
    category = st.selectbox("Category", ["Groceries", "Dining", "Entertainment", "Transportation", "Shopping", "Utilities", "Housing"])
    amount = st.number_input("Monthly Budget Amount", min_value=0.0, value=100.0, step=10.0)
    start_date = st.date_input("Start Date")
    
    submit_button = st.form_submit_button("Create Budget Goal")
    if submit_button:
        st.success(f"Budget goal created for {category}: ${amount:.2f}")

# Progress visualization
st.markdown("### Budget Progress")
for idx, row in budget_data.iterrows():
    category = row["Category"]
    budget = row["Monthly Budget"]
    spent = row["Current Spending"]
    percentage = int((spent / budget) * 100)
    
    st.markdown(f"**{category}**")
    st.progress(percentage / 100)
    st.text(f"${spent:.2f} of ${budget:.2f} ({percentage}%)")
    st.markdown("---")
