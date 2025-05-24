"""
FinSight - Dashboard Page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Dashboard - FinSight",
    page_icon="📊",
    layout="wide"
)

st.title("Dashboard 📊")
st.subheader("Financial Overview")

# Path for transactions file
DATA_DIR = "data"
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.csv")

# Helper function to load transactions
def load_transactions():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if os.path.exists(TRANSACTIONS_FILE):
        return pd.read_csv(TRANSACTIONS_FILE)
    else:
        return pd.DataFrame({
            "date": [],
            "amount": [],
            "currency": [],
            "category": [],
            "description": [],
            "type": []  # 'income' or 'expense'
        })

# Load transactions
transactions = load_transactions()

# If no transactions exist, show sample/placeholder content
if transactions.empty:
    st.info("No transactions found. Add some transactions to see your financial overview.")
    
    # Mock data for demonstration
    st.markdown("### Sample Dashboard View")
    
    # Sample income vs expense chart
    sample_data = {
        "Month": ["Jan", "Feb", "Mar", "Apr", "May"],
        "Income": [3000, 3200, 3100, 3300, 3500],
        "Expense": [2700, 2900, 2800, 3000, 3100]
    }
    sample_df = pd.DataFrame(sample_data)
    
    # Create a bar chart
    st.bar_chart(sample_df.set_index("Month"))
    
    # Sample transaction table
    st.markdown("### Sample Recent Transactions")
    sample_transactions = pd.DataFrame({
        "Date": ["2023-11-01", "2023-11-03", "2023-11-05", "2023-11-10", "2023-11-15"],
        "Description": ["Grocery Store", "Electricity Bill", "Restaurant", "Gas Station", "Salary"],
        "Amount": ["- KZT 120.45", "- KZT 85.20", "- KZT 65.30", "- KZT 45.67", "+ KZT 1,500.00"],
        "Category": ["Groceries", "Utilities", "Dining", "Transportation", "Income"],
        "Type": ["Expense", "Expense", "Expense", "Expense", "Income"]
    })
    st.dataframe(sample_transactions)
    
    # Add a link to the Add Transaction page
    st.markdown("[Add your first transaction →](Add_Transaction)")
else:
    # Process transactions for dashboard
    transactions["date"] = pd.to_datetime(transactions["date"])
    
    # Get the default currency (most commonly used)
    default_currency = transactions["currency"].mode()[0]
    
    # Filter by time period
    time_period = st.selectbox(
        "Time Period",
        ["Last 30 Days", "Last 3 Months", "Last 6 Months", "Year to Date", "All Time"],
        index=0
    )
    
    today = datetime.now().date()
    
    if time_period == "Last 30 Days":
        start_date = today - timedelta(days=30)
    elif time_period == "Last 3 Months":
        start_date = today - timedelta(days=90)
    elif time_period == "Last 6 Months":
        start_date = today - timedelta(days=180)
    elif time_period == "Year to Date":
        start_date = datetime(today.year, 1, 1).date()
    else:  # All Time
        start_date = transactions["date"].min().date()
    
    filtered_transactions = transactions[transactions["date"] >= pd.Timestamp(start_date)]
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    # Total Income
    total_income = filtered_transactions[filtered_transactions["type"] == "income"]["amount"].sum()
    
    # Total Expenses
    total_expenses = filtered_transactions[filtered_transactions["type"] == "expense"]["amount"].sum()
    
    # Net Balance
    net_balance = total_income - total_expenses
    
    with col1:
        st.metric(
            "Total Income", 
            f"{default_currency} {total_income:,.2f}",
            delta=None
        )
    
    with col2:
        st.metric(
            "Total Expenses", 
            f"{default_currency} {total_expenses:,.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            "Net Balance", 
            f"{default_currency} {net_balance:,.2f}",
            delta=None,
            delta_color="normal" if net_balance >= 0 else "inverse"
        )
    
    # Income vs Expenses by Month
    st.markdown("### Income vs Expenses")
    
    # Group by month and transaction type
    filtered_transactions["month"] = filtered_transactions["date"].dt.strftime("%b %Y")
    monthly_summary = filtered_transactions.groupby(["month", "type"])["amount"].sum().reset_index()
    
    # Pivot to get income and expenses as columns
    monthly_pivot = monthly_summary.pivot(index="month", columns="type", values="amount").reset_index()
    monthly_pivot = monthly_pivot.fillna(0)
    
    # Ensure both income and expense columns exist
    if "income" not in monthly_pivot.columns:
        monthly_pivot["income"] = 0
    if "expense" not in monthly_pivot.columns:
        monthly_pivot["expense"] = 0
    
    # Create a bar chart
    fig = px.bar(
        monthly_pivot,
        x="month",
        y=["income", "expense"],
        barmode="group",
        labels={"value": "Amount", "month": "Month", "variable": "Type"},
        color_discrete_map={"income": "green", "expense": "red"},
        title="Monthly Income vs Expenses"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top Spending Categories
    st.markdown("### Top Spending Categories")
    expense_by_category = filtered_transactions[filtered_transactions["type"] == "expense"].groupby("category")["amount"].sum().reset_index()
    expense_by_category = expense_by_category.sort_values("amount", ascending=False).head(5)
    
    if not expense_by_category.empty:
        fig = px.pie(
            expense_by_category,
            values="amount",
            names="category",
            title="Top Expense Categories",
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No expense data available for the selected period.")
    
    # Recent Transactions
    st.markdown("### Recent Transactions")
    recent_transactions = filtered_transactions.sort_values("date", ascending=False).head(5)
    
    if not recent_transactions.empty:
        # Format for display
        display_df = recent_transactions.copy()
        
        # Format amount with currency and color
        def format_amount(row):
            amount = row["amount"]
            currency = row["currency"]
            transaction_type = row["type"]
            
            if transaction_type == "expense":
                return f"<span style='color:red'>-{currency} {abs(float(amount)):.2f}</span>"
            else:  # income
                return f"<span style='color:green'>+{currency} {abs(float(amount)):.2f}</span>"
        
        display_df["formatted_amount"] = display_df.apply(format_amount, axis=1)
        
        # Format date
        display_df["formatted_date"] = display_df["date"].dt.strftime("%Y-%m-%d")
        
        # Select and rename columns for display
        display_df = display_df[["formatted_date", "category", "formatted_amount", "description"]]
        display_df.columns = ["Date", "Category", "Amount", "Description"]
        
        # Display with HTML formatting
        st.write(display_df.to_html(escape=False), unsafe_allow_html=True)
        
        # Link to all transactions
        st.markdown("[View all transactions →](Transactions)")
    else:
        st.info("No transactions available for the selected period.")
