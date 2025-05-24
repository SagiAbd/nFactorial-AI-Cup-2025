"""
FinSight - Transactions Page
"""
import streamlit as st
import pandas as pd
import os

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
        required_columns = ["date", "amount", "currency", "category", "type"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
        else:
            # Add description column if not exists
            if "description" not in df.columns:
                df["description"] = ""
                
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
            (pd.to_datetime(filtered_df["date"]) >= pd.to_datetime(start_date)) & 
            (pd.to_datetime(filtered_df["date"]) <= pd.to_datetime(end_date))
        ]
    
    # Display transactions with colored amounts based on type
    if not filtered_df.empty:
        # Format the display
        display_df = filtered_df.copy()
        
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
        
        # Reorder and select columns for display
        cols_to_display = ["date", "formatted_amount", "category", "description", "type"]
        display_df = display_df[cols_to_display]
        
        # Rename columns for display
        display_df = display_df.rename(columns={
            "date": "Date",
            "formatted_amount": "Amount",
            "category": "Category",
            "description": "Description",
            "type": "Type"
        })
        
        # Display with HTML formatting
        st.write(display_df.to_html(escape=False), unsafe_allow_html=True)
        
        # Summary statistics
        st.markdown("### Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            total_income = filtered_df[filtered_df["type"] == "income"]["amount"].sum()
            st.metric("Total Income", f"{filtered_df['currency'].iloc[0]} {total_income:.2f}")
        
        with col2:
            total_expense = filtered_df[filtered_df["type"] == "expense"]["amount"].sum()
            st.metric("Total Expenses", f"{filtered_df['currency'].iloc[0]} {total_expense:.2f}")
        
        # Net balance
        st.metric("Net Balance", f"{filtered_df['currency'].iloc[0]} {(total_income - total_expense):.2f}")
    else:
        st.info("No transactions match the selected filters.")
else:
    # Show sample data if no transactions exist
    st.markdown("### Sample Transactions")
    sample_transactions = pd.DataFrame({
        "date": ["2023-11-01", "2023-11-03", "2023-11-05", "2023-11-07", "2023-11-10"],
        "amount": [120.45, 85.20, 65.30, 45.67, 1500.00],
        "currency": ["KZT", "KZT", "KZT", "KZT", "KZT"],
        "category": ["Groceries", "Utilities", "Dining", "Transportation", "Salary"],
        "description": ["Weekly shopping", "Electricity bill", "Restaurant dinner", "Gas station", "Monthly salary"],
        "type": ["expense", "expense", "expense", "expense", "income"]
    })
    
    # Display sample data
    st.dataframe(sample_transactions)
    
    st.info("You can add transactions using the 'Add Transaction' page or upload a CSV file above.")
