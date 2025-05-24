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
            "datetime": [],
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
        required_columns = ["datetime", "amount", "currency", "category", "type"]
        # Handle legacy files with 'date' column
        if "date" in df.columns and "datetime" not in df.columns:
            df["datetime"] = df["date"]
            df = df.drop("date", axis=1)
            required_columns = ["datetime" if col == "date" else col for col in required_columns]
            
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
            (pd.to_datetime(filtered_df["datetime"]) >= pd.to_datetime(start_date)) & 
            (pd.to_datetime(filtered_df["datetime"]) <= pd.to_datetime(end_date))
        ]
    
    # Display filtered transactions
    if not filtered_df.empty:
        # Format dates for display
        filtered_df["formatted_date"] = pd.to_datetime(filtered_df["datetime"]).dt.strftime('%Y-%m-%d %H:%M')
        
        # Determine columns to display
        display_df = filtered_df[["formatted_date", "type", "category", "amount", "currency", "description"]]
        
        # Set column names and reorder
        display_df.columns = ["Date & Time", "Type", "Category", "Amount", "Currency", "Description"]
        
        # Style the dataframe
        st.dataframe(
            display_df.style.applymap(
                lambda x: "color: green" if x == "income" else "color: red", 
                subset=["Type"]
            ),
            use_container_width=True
        )
        
        # Add download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
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
    st.markdown("""
    ### Required Format for CSV Uploads
    
    Your CSV file should include the following columns:
    
    - `datetime` - Transaction date and time (YYYY-MM-DD HH:MM:SS format)
    - `amount` - Transaction amount (positive for income, negative for expenses)
    - `currency` - Currency code (e.g., KZT, USD)
    - `category` - Transaction category
    - `type` - Transaction type ('income' or 'expense')
    
    Optional columns:
    - `description` - Transaction description or notes
    
    #### Sample CSV Format:
    ```
    datetime,amount,currency,category,type,description
    2023-05-01 14:30:00,50000,KZT,Salary,income,Monthly salary
    2023-05-02 12:15:00,-2500,KZT,Food,expense,Lunch
    ```
    """)
