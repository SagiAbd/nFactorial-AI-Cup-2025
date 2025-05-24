"""
Finance Agent App - Reports Page
"""
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Reports - Finance Agent",
    page_icon="📈",
    layout="wide"
)

st.title("Reports 📈")
st.subheader("Financial Reports and Insights")

# Time period selector
st.markdown("### Select Time Period")
report_period = st.selectbox("Report Period", ["Last Month", "Last 3 Months", "Last 6 Months", "Year to Date", "Custom"])

if report_period == "Custom":
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")

# Report types
report_type = st.radio(
    "Report Type",
    ["Spending by Category", "Income vs. Expenses", "Monthly Trends", "Savings Analysis"]
)

# Placeholder for different report types
if report_type == "Spending by Category":
    st.markdown("### Spending by Category")
    
    # Mock data for demonstration
    categories = ["Groceries", "Dining", "Entertainment", "Transportation", "Shopping", "Utilities"]
    values = [420, 250, 150, 180, 320, 210]
    
    # Display chart
    chart_data = pd.DataFrame({
        "Category": categories,
        "Amount": values
    })
    st.bar_chart(chart_data.set_index("Category"))
    
    # Display table
    st.dataframe(chart_data)

elif report_type == "Income vs. Expenses":
    st.markdown("### Income vs. Expenses")
    
    # Mock data for demonstration
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    income = [3000, 3000, 3200, 3200, 3200, 3500]
    expenses = [2700, 2500, 2800, 2600, 2900, 2700]
    
    chart_data = pd.DataFrame({
        "Month": months,
        "Income": income,
        "Expenses": expenses
    })
    
    st.line_chart(chart_data.set_index("Month"))
    
    # Calculate savings rate
    savings = sum(income) - sum(expenses)
    savings_rate = (savings / sum(income)) * 100
    
    st.metric("Total Savings", f"${savings:.2f}", f"{savings_rate:.1f}% of income")

elif report_type == "Monthly Trends":
    st.markdown("### Monthly Spending Trends")
    
    # Mock data for demonstration
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    data = np.random.randn(6, 5) * 100 + 300
    chart_data = pd.DataFrame(
        data,
        columns=["Groceries", "Dining", "Entertainment", "Transportation", "Shopping"]
    )
    chart_data.insert(0, "Month", months)
    
    st.line_chart(chart_data.set_index("Month"))

elif report_type == "Savings Analysis":
    st.markdown("### Savings Analysis")
    
    # Mock data for demonstration
    st.metric("Current Savings Rate", "15%", "2% from last month")
    st.metric("Projected Annual Savings", "$5,400", "$400 more than last year")
    
    st.info("This report will provide detailed analysis of your savings patterns and recommendations for improvement.")
