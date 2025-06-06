"""
FinSight - Dashboard Page
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from datetime import date, timedelta
import sys

# Add parent directory to path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.currency_utils import convert_to_kzt, format_currency, CURRENCY_SYMBOLS

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
        "Amount": ["- ₸ 120.45", "- ₸ 85.20", "- ₸ 65.30", "- ₸ 45.67", "+ ₸ 1,500.00"],
        "Category": ["Groceries", "Utilities", "Dining", "Transportation", "Income"],
        "Type": ["Expense", "Expense", "Expense", "Expense", "Income"]
    })
    st.dataframe(sample_transactions)
    
    # Add a link to the Add Transaction page
    st.markdown("[Add your first transaction →](Add_Transaction)")
else:
    # Data preprocessing
    if not transactions.empty:
        # Ensure date column is properly formatted
        transactions["date"] = pd.to_datetime(transactions["date"], format='%Y-%m-%d', errors='coerce')
        
        # Convert all amounts to KZT
        transactions["amount_kzt"] = transactions.apply(
            lambda row: convert_to_kzt(row["amount"], row["currency"]), 
            axis=1
        )
        
        # Filter data based on time period
        time_period = st.sidebar.selectbox(
            "Time Period",
            ["Last 30 Days", "Last 3 Months", "Last 6 Months", "Year to Date", "All Time", "Custom Range"]
        )
        
        # Determine start date based on selected time period
        if time_period == "Last 30 Days":
            start_date = (date.today() - timedelta(days=30))
        elif time_period == "Last 3 Months":
            start_date = (date.today() - timedelta(days=90))
        elif time_period == "Last 6 Months":
            start_date = (date.today() - timedelta(days=180))
        elif time_period == "Year to Date":
            start_date = date(date.today().year, 1, 1)
        elif time_period == "All Time":
            start_date = transactions["date"].min().date()
        else:  # Custom Range
            date_range = st.sidebar.date_input(
                "Select Date Range",
                value=[transactions["date"].min().date(), date.today()],
                min_value=transactions["date"].min().date(),
                max_value=date.today()
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
                filtered_transactions = transactions[
                    (transactions["date"] >= pd.Timestamp(start_date)) & 
                    (transactions["date"] <= pd.Timestamp(end_date))
                ]
            else:
                start_date = transactions["date"].min().date()
                filtered_transactions = transactions[transactions["date"] >= pd.Timestamp(start_date)]
        
        # Apply date filter if not custom range
        if time_period != "Custom Range":
            filtered_transactions = transactions[transactions["date"] >= pd.Timestamp(start_date)]
        
        # Calculate overview metrics using KZT values
        total_income = filtered_transactions[filtered_transactions["amount_kzt"] > 0]["amount_kzt"].sum()
        total_expenses = abs(filtered_transactions[filtered_transactions["amount_kzt"] < 0]["amount_kzt"].sum())
        net_flow = total_income - total_expenses
        
        # Display metrics with KZT symbol
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Income", f"{CURRENCY_SYMBOLS['KZT']} {total_income:,.2f}")
        with col2:
            st.metric("Total Expenses", f"{CURRENCY_SYMBOLS['KZT']} {total_expenses:,.2f}")
        with col3:
            delta_color = "normal" if net_flow >= 0 else "inverse"
            st.metric("Net Cash Flow", f"{CURRENCY_SYMBOLS['KZT']} {net_flow:,.2f}", 
                     delta=f"{CURRENCY_SYMBOLS['KZT']} {net_flow:,.2f}", 
                     delta_color=delta_color)
        
        # Monthly trend analysis
        st.subheader("Monthly Cash Flow")
        
        # Prepare data for monthly trend
        filtered_transactions["month"] = filtered_transactions["date"].dt.strftime("%b %Y")
        
        monthly_income = filtered_transactions[filtered_transactions["amount_kzt"] > 0].groupby("month")["amount_kzt"].sum()
        monthly_expenses = abs(filtered_transactions[filtered_transactions["amount_kzt"] < 0].groupby("month")["amount_kzt"].sum())
        
        # Create dataframe for plotting
        months = sorted(filtered_transactions["month"].unique(), key=lambda x: pd.to_datetime(x, format="%b %Y"))
        
        # Income vs Expenses by Month
        st.markdown("### Income vs Expenses")
        
        # Group by month and transaction type
        filtered_transactions["month"] = filtered_transactions["date"].dt.strftime("%b %Y")
        monthly_summary = filtered_transactions.groupby(["month", "type"])["amount_kzt"].sum().reset_index()
        
        # Pivot to get income and expenses as columns
        monthly_pivot = monthly_summary.pivot(index="month", columns="type", values="amount_kzt").reset_index()
        monthly_pivot = monthly_pivot.fillna(0)
        
        # Ensure both income and expense columns exist
        if "income" not in monthly_pivot.columns:
            monthly_pivot["income"] = 0
        if "expense" not in monthly_pivot.columns:
            monthly_pivot["expense"] = 0
            
        # Make expense values positive for better visualization
        monthly_pivot["expense"] = monthly_pivot["expense"].abs()
        
        # Create a bar chart
        fig = px.bar(
            monthly_pivot,
            x="month",
            y=["income", "expense"],
            barmode="group",
            labels={"value": f"Amount ({CURRENCY_SYMBOLS['KZT']})", "month": "Month", "variable": "Type"},
            color_discrete_map={"income": "green", "expense": "red"},
            title=f"Monthly Income vs Expenses ({CURRENCY_SYMBOLS['KZT']})"
        )
        
        # Update y-axis to show KZT formatting
        fig.update_layout(
            yaxis=dict(
                tickprefix=f"{CURRENCY_SYMBOLS['KZT']} ",
                separatethousands=True
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Top Spending Categories
        st.markdown("### Top Spending Categories")
        expense_by_category = filtered_transactions[filtered_transactions["type"] == "expense"].groupby("category")["amount_kzt"].sum().abs().reset_index()
        expense_by_category = expense_by_category.sort_values("amount_kzt", ascending=False).head(5)
        
        if not expense_by_category.empty:
            fig = px.pie(
                expense_by_category,
                values="amount_kzt",
                names="category",
                title=f"Top Expense Categories ({CURRENCY_SYMBOLS['KZT']})",
                hole=0.4
            )
            # Add KZT formatting to hover text
            fig.update_traces(
                hovertemplate='<b>%{label}</b><br>Amount: ' + CURRENCY_SYMBOLS['KZT'] + ' %{value:,.2f}<br>Percent: %{percent}'
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
            
            # Format amount with currency symbol and color
            def format_amount_display(row):
                # Format original amount with original currency
                original_formatted = format_currency(
                    row["amount"], 
                    row["currency"], 
                    include_symbol=True, 
                    colorize=True, 
                    transaction_type=row["type"]
                )
                
                # If currency is not KZT, also show KZT equivalent
                if row["currency"] != "KZT":
                    kzt_formatted = format_currency(
                        row["amount_kzt"], 
                        "KZT", 
                        include_symbol=True, 
                        colorize=False
                    )
                    return f"{original_formatted}<br><small>({kzt_formatted})</small>"
                
                return original_formatted
            
            display_df["formatted_amount"] = display_df.apply(format_amount_display, axis=1)
            
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
