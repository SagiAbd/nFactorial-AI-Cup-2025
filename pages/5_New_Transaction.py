"""
FinSight - Add Transaction Page
"""
import streamlit as st
import pandas as pd
from datetime import datetime, time, date
import json
import os

st.set_page_config(
    page_title="Add Transaction - FinSight",
    page_icon="💲",
    layout="wide"
)

st.title("Add Transaction 💲")
st.subheader("Record your income and expenses")

# Define default categories with icons
EXPENSE_CATEGORIES = {
    "Food & Dining": "🍽️",
    "Groceries": "🛒",
    "Transportation": "🚗",
    "Shopping": "🛍️",
    "Entertainment": "🎬",
    "Utilities": "💡",
    "Housing": "🏠",
    "Healthcare": "🏥",
    "Education": "📚",
    "Personal Care": "💇",
    "Travel": "✈️",
    "Gifts & Donations": "🎁",
    "Other": "📋"
}

INCOME_CATEGORIES = {
    "Salary": "💼",
    "Freelance": "💻",
    "Business": "🏢",
    "Investments": "📈",
    "Gifts": "🎁",
    "Rental Income": "🏘️",
    "Other": "📋"
}

# Path for storing transactions
DATA_DIR = "data"
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.csv")

# Helper function to load existing transactions
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

# Helper function to save transactions
def save_transaction(transaction_datetime, amount, currency, category, description, transaction_type):
    df = load_transactions()
    
    new_transaction = pd.DataFrame({
        "datetime": [transaction_datetime.strftime('%Y-%m-%d %H:%M:%S')],
        "amount": [amount],
        "currency": [currency],
        "category": [category],
        "description": [description],
        "type": [transaction_type]
    })
    
    df = pd.concat([df, new_transaction], ignore_index=True)
    df.to_csv(TRANSACTIONS_FILE, index=False)
    return df

# Transaction type selector
transaction_type = st.radio(
    "Transaction Type",
    ["Expense", "Income"],
    horizontal=True
)

# Form for adding transaction
with st.form("transaction_form"):
    # Amount field
    amount = st.number_input("Amount", min_value=0.01, step=1.0, format="%.2f")
    
    # Currency selector
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write("Currency")
    with col2:
        currency = st.selectbox(
            "Select Currency",
            ["KZT", "USD", "EUR", "RUB"],
            label_visibility="collapsed"
        )
    
    # Date input (without time)
    transaction_date = st.date_input("Date", datetime.now().date())
    
    # Determine time based on selected date
    current_date = datetime.now().date()
    current_time = datetime.now().time()
    
    # If user selects today's date, use current time
    # Otherwise, use noon (12:00) as default time
    if transaction_date == current_date:
        transaction_time = current_time
    else:
        # Default time for non-current dates (noon - 12:00)
        transaction_time = time(12, 0, 0)
    
    # Show the selected time (read-only info)
    time_str = transaction_time.strftime("%H:%M:%S")
    st.caption(f"Time: {time_str}")
    
    # Combine date and time into a datetime object
    transaction_datetime = datetime.combine(transaction_date, transaction_time)
    
    # Category selection based on transaction type
    if transaction_type == "Expense":
        categories = EXPENSE_CATEGORIES
    else:  # Income
        categories = INCOME_CATEGORIES
    
    # Category display with icons
    st.write("Category")
    
    # Create a grid of category buttons
    cols = st.columns(4)
    custom_category = None
    
    # First show default categories as buttons
    category_list = list(categories.keys())
    selected_category = st.session_state.get("selected_category", None)
    
    # Category buttons in a 4-column grid
    for i, (category, icon) in enumerate(categories.items()):
        col_idx = i % 4
        with cols[col_idx]:
            button_label = f"{icon} {category}"
            if st.button(button_label, key=f"cat_{i}", 
                       type="primary" if selected_category == category else "secondary"):
                selected_category = category
                st.session_state["selected_category"] = category
    
    # Option to add custom category
    st.write("Or add a custom category")
    col1, col2 = st.columns([3, 1])
    with col1:
        custom_name = st.text_input("Category name", key="custom_cat_name")
    with col2:
        custom_icon = st.text_input("Icon", placeholder="🏷️", key="custom_cat_icon")
    
    # Description field
    description = st.text_area("Description (optional)")
    
    # Submit button
    submitted = st.form_submit_button("Add Transaction")

# Handle form submission
if submitted:
    if amount <= 0:
        st.error("Amount must be greater than zero")
    elif not currency:
        st.error("Please select a currency")
    elif not selected_category and not custom_name:
        st.error("Please select or add a category")
    else:
        # Use custom category if provided
        final_category = None
        if custom_name:
            if not custom_icon:
                custom_icon = "📋"  # Default icon
            final_category = custom_name
            # Add to session state for future use
            if transaction_type == "Expense":
                EXPENSE_CATEGORIES[custom_name] = custom_icon
            else:
                INCOME_CATEGORIES[custom_name] = custom_icon
        else:
            final_category = selected_category
        
        # Save transaction
        transaction_df = save_transaction(
            transaction_datetime=transaction_datetime,
            amount=amount,
            currency=currency,
            category=final_category,
            description=description,
            transaction_type=transaction_type.lower()
        )
        
        st.success(f"Transaction added: {'+' if transaction_type == 'Income' else '-'}{amount} {currency}")
        
        # Show latest transactions
        st.subheader("Recent Transactions")
        recent_transactions = transaction_df.sort_values("datetime", ascending=False).head(5)
        
        # Format for display
        for _, tx in recent_transactions.iterrows():
            amount_color = "green" if tx["type"] == "income" else "red"
            amount_sign = "+" if tx["type"] == "income" else "-"
            
            st.markdown(
                f"**{pd.to_datetime(tx['datetime']).strftime('%Y-%m-%d %H:%M')}** | "
                f"**:{amount_color}[{amount_sign}{abs(float(tx['amount'])):.2f} {tx['currency']}]** | "
                f"📂 {tx['category']} | "
                f"📝 {tx['description'] if not pd.isna(tx['description']) else 'No description'}"
            )
            st.markdown("---") 