"""
FinSight - Add Transaction Page
"""
import streamlit as st
import pandas as pd
from datetime import datetime
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
            "date": [],
            "amount": [],
            "currency": [],
            "category": [],
            "description": [],
            "type": []  # 'income' or 'expense'
        })

# Helper function to save transactions
def save_transaction(date, amount, currency, category, description, transaction_type):
    df = load_transactions()
    
    new_transaction = pd.DataFrame({
        "date": [date],
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
    
    # Date picker
    date = st.date_input("Date", datetime.now().date())
    
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
        custom_icon = st.selectbox("Icon", ["💰", "🎮", "🎵", "🚀", "🏆", "🎨", "🎭", "🍕", "🍺", "☕"], label_visibility="collapsed")
    
    if custom_name and custom_icon:
        if st.button(f"Add {custom_icon} {custom_name} as category"):
            # In a real app, you would save this to a user preferences file
            if transaction_type == "Expense":
                EXPENSE_CATEGORIES[custom_name] = custom_icon
            else:
                INCOME_CATEGORIES[custom_name] = custom_icon
            selected_category = custom_name
            st.session_state["selected_category"] = custom_name
    
    # Show the currently selected category
    if selected_category:
        icon = categories.get(selected_category, "📋")
        st.success(f"Selected category: {icon} {selected_category}")
    
    # Description field
    description = st.text_area("Description (optional)", height=100)
    
    # Submit button
    submitted = st.form_submit_button("Add Transaction")
    
    if submitted:
        if not selected_category:
            st.error("Please select a category")
        elif amount <= 0:
            st.error("Please enter a valid amount")
        else:
            # Save the transaction
            transaction_df = save_transaction(
                date.strftime("%Y-%m-%d"), 
                amount, 
                currency, 
                selected_category, 
                description, 
                transaction_type.lower()
            )
            
            # Success message with details
            st.success(f"Added {transaction_type.lower()}: {currency} {amount:.2f} in {selected_category}")
            
            # Show updated transactions
            st.write("Recent Transactions:")
            st.dataframe(transaction_df.tail(5), use_container_width=True)
            
            # Clear the form
            st.session_state["selected_category"] = None 