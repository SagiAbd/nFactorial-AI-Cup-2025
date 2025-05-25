"""
FinSight - Transactions Page
"""
import streamlit as st
import pandas as pd
import os
import sys

# Add parent directory to path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.currency_utils import convert_to_kzt, format_currency, CURRENCY_SYMBOLS

st.set_page_config(
    page_title="Transactions - FinSight",
    page_icon="💳",
    layout="wide"
)

st.title("Transactions 💳")
st.subheader("View and Manage Your Transactions")

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

# File uploader for transaction data
st.markdown("### Upload Transaction Data")
uploaded_file = st.file_uploader("Upload your transaction CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read uploaded file
        df = pd.read_csv(uploaded_file)
        
        # Check if the file has the required columns
        required_columns = ["date", "amount", "currency", "type", "category"]
        # Handle legacy files with 'datetime' column
        if "datetime" in df.columns and "date" not in df.columns:
            df["date"] = df["datetime"]
            df = df.drop("datetime", axis=1)
            required_columns = ["date" if col == "datetime" else col for col in required_columns]
            
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            # Add description column if not exists
            if "description" not in df.columns:
                df["description"] = ""
                
            # Ensure proper column order
            df = df[['date', 'description', 'amount', 'type', 'category', 'currency']]
                
            # Save the transactions
            if not os.path.exists(DATA_DIR):
                os.makedirs(DATA_DIR)
            df.to_csv(TRANSACTIONS_FILE, index=False)
            st.success("Transactions imported successfully!")
    except Exception as e:
        st.error(f"Error processing file: {e}")
        
# Load and display transactions
transactions = load_transactions()

if not transactions.empty:
    # Convert all amounts to KZT for analytics
    transactions["amount_kzt"] = transactions.apply(
        lambda row: convert_to_kzt(row["amount"], row["currency"]), 
        axis=1
    )
    
    st.markdown("### Your Transactions")
    
    # Add filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Transaction type filter
        type_filter = st.selectbox(
            "Transaction Type", 
            ["All", "Income", "Expense"]
        )
    
    with col2:
        # Category filter
        all_categories = transactions["category"].unique().tolist()
        category_filter = st.selectbox(
            "Category", 
            ["All"] + all_categories
        )
    
    with col3:
        # Date range filter
        date_range = st.date_input(
            "Date Range",
            value=[],
            max_value=pd.to_datetime("today")
        )
    
    # Apply filters
    filtered_df = transactions.copy()
    
    if type_filter != "All":
        filtered_df = filtered_df[filtered_df["type"] == type_filter.lower()]
    
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df["category"] == category_filter]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (pd.to_datetime(filtered_df["date"], format='%Y-%m-%d', errors='coerce') >= pd.to_datetime(start_date)) & 
            (pd.to_datetime(filtered_df["date"], format='%Y-%m-%d', errors='coerce') <= pd.to_datetime(end_date))
        ]
    
    # Display filtered transactions
    if not filtered_df.empty:
        # Format dates for display - Note: dates are now YYYY-MM-DD without time
        filtered_df["formatted_date"] = pd.to_datetime(filtered_df["date"], format='%Y-%m-%d', errors='coerce').dt.strftime('%Y-%m-%d')
        
        # Format amount with currency symbol and KZT equivalent
        filtered_df["formatted_amount"] = filtered_df.apply(
            lambda row: format_currency(row["amount"], row["currency"], include_symbol=True, colorize=True, transaction_type=row["type"]), 
            axis=1
        )
        
        # Format KZT amount for display
        filtered_df["amount_kzt_formatted"] = filtered_df.apply(
            lambda row: format_currency(row["amount_kzt"], "KZT", include_symbol=True), 
            axis=1
        )
        
        # Display statistics in KZT
        if len(filtered_df) > 0:
            st.markdown("### Transaction Summary (in KZT)")
            
            total_income_kzt = filtered_df[filtered_df["amount_kzt"] > 0]["amount_kzt"].sum()
            total_expense_kzt = abs(filtered_df[filtered_df["amount_kzt"] < 0]["amount_kzt"].sum())
            net_flow_kzt = total_income_kzt - total_expense_kzt
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.metric("Total Income", f"{CURRENCY_SYMBOLS['KZT']} {total_income_kzt:,.2f}")
            
            with stat_col2:
                st.metric("Total Expenses", f"{CURRENCY_SYMBOLS['KZT']} {total_expense_kzt:,.2f}")
            
            with stat_col3:
                delta_color = "normal" if net_flow_kzt >= 0 else "inverse"
                st.metric("Net Cash Flow", f"{CURRENCY_SYMBOLS['KZT']} {net_flow_kzt:,.2f}", 
                         delta=f"{CURRENCY_SYMBOLS['KZT']} {net_flow_kzt:,.2f}", 
                         delta_color=delta_color)
        
        # Combine formatted amount with KZT equivalent for non-KZT transactions
        def display_amount(row):
            if row["currency"] != "KZT":
                return f"{row['formatted_amount']}<br><small>({row['amount_kzt_formatted']})</small>"
            return row["formatted_amount"]
        
        filtered_df["display_amount"] = filtered_df.apply(display_amount, axis=1)
        
        # Determine columns to display
        display_df = filtered_df[["formatted_date", "description", "display_amount", "type", "category"]]
        
        # Set column names and reorder
        display_df.columns = ["Date", "Description", "Amount", "Type", "Category"]
        
        # Display with HTML formatting
        st.write(display_df.to_html(escape=False), unsafe_allow_html=True)
        
        # Add download button
        csv = filtered_df[["date", "description", "amount", "currency", "type", "category", "amount_kzt"]].to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download Filtered Data", 
            csv, 
            "transactions_export.csv", 
            "text/csv", 
            key="download-csv"
        )
    else:
        st.info("No transactions match your filter criteria.")
else:
    st.info("No transactions found. Add some transactions or import a CSV file.")

# Display upload instructions
with st.expander("CSV Upload Format Instructions"):
    st.markdown(f"""
    ### Required Format for CSV Uploads
    
    Your CSV file should include the following columns:
    
    - `date` - Transaction date (YYYY-MM-DD format)
    - `description` - Transaction description or notes
    - `amount` - Transaction amount (positive for income, negative for expenses)
    - `currency` - Currency code (e.g., {CURRENCY_SYMBOLS['KZT']} KZT, $ USD)
    - `type` - Transaction type ('income' or 'expense')
    - `category` - Transaction category
    
    #### Sample CSV Format:
    ```
    date,description,amount,currency,type,category
    2023-05-01,Monthly salary,50000,KZT,income,Salary
    2023-05-02,Lunch,-2500,KZT,expense,Food
    ```
    
    > **Note:** All non-KZT currencies will be automatically converted to KZT ({CURRENCY_SYMBOLS['KZT']}) for analytics.
    """)
