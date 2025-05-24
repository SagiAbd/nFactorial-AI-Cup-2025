"""
FinSight - Budget Goals Page
"""
import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Budget Goals - FinSight",
    page_icon="🎯",
    layout="wide"
)

st.title("Budget Goals 🎯")
st.subheader("Set and Track Your Financial Goals")

# Paths for data files
DATA_DIR = "data"
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.csv")
BUDGET_GOALS_FILE = os.path.join(DATA_DIR, "budget_goals.json")

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
            "type": []
        })

# Helper function to load budget goals
def load_budget_goals():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    if os.path.exists(BUDGET_GOALS_FILE):
        with open(BUDGET_GOALS_FILE, 'r') as f:
            return json.load(f)
    else:
        return {}

# Helper function to save budget goals
def save_budget_goals(budget_goals):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    with open(BUDGET_GOALS_FILE, 'w') as f:
        json.dump(budget_goals, f)

# Load transactions and budget goals
transactions = load_transactions()
budget_goals = load_budget_goals()

# Current month
current_month = datetime.now().strftime("%Y-%m")
current_month_name = datetime.now().strftime("%B %Y")

# Define default expense categories from the Add Transaction page
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

# If transactions exist, get actual categories from transactions
if not transactions.empty:
    expense_categories = transactions[transactions["type"] == "expense"]["category"].unique().tolist()
    # Use both actual categories and default categories
    all_categories = sorted(list(set(expense_categories + list(EXPENSE_CATEGORIES.keys()))))
else:
    all_categories = sorted(list(EXPENSE_CATEGORIES.keys()))

# Currency selection
default_currency = "KZT"
if not transactions.empty and "currency" in transactions.columns:
    default_currency = transactions["currency"].mode()[0]

# Budget creation form
st.markdown("### Create New Budget Goal")
with st.form("budget_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox("Category", all_categories)
    
    with col2:
        currency = st.selectbox("Currency", ["KZT", "USD", "EUR", "RUB"], index=["KZT", "USD", "EUR", "RUB"].index(default_currency))
    
    amount = st.number_input("Monthly Budget Amount", min_value=0.0, value=100.0, step=10.0)
    
    # Date range for the budget goal
    col1, col2 = st.columns(2)
    with col1:
        start_month = st.date_input("Start Month", value=datetime.now().replace(day=1)).strftime("%Y-%m")
    with col2:
        end_month = st.date_input("End Month", value=(datetime.now() + timedelta(days=90)).replace(day=1)).strftime("%Y-%m")
    
    submit_button = st.form_submit_button("Create Budget Goal")
    
    if submit_button:
        if category and amount > 0:
            # Create or update budget goal
            if current_month not in budget_goals:
                budget_goals[current_month] = {}
            
            budget_goals[current_month][category] = {
                "amount": amount,
                "currency": currency,
                "start_month": start_month,
                "end_month": end_month
            }
            
            save_budget_goals(budget_goals)
            st.success(f"Budget goal created for {category}: {currency} {amount:.2f}")
        else:
            st.error("Please select a category and enter an amount greater than zero.")

# Display current month's budget goals
st.markdown(f"### Your Budget Goals for {current_month_name}")

if current_month in budget_goals and budget_goals[current_month]:
    # Create dataframe for budget goals
    budget_data = []
    
    # Calculate current spending for each category
    current_spending = {}
    
    if not transactions.empty:
        # Convert date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(transactions["date"]):
            transactions["date"] = pd.to_datetime(transactions["date"])
        
        # Filter transactions for current month
        current_month_start = datetime.strptime(current_month, "%Y-%m").replace(day=1)
        next_month_start = (current_month_start + timedelta(days=32)).replace(day=1)
        
        month_transactions = transactions[
            (transactions["date"] >= pd.Timestamp(current_month_start)) & 
            (transactions["date"] < pd.Timestamp(next_month_start)) &
            (transactions["type"] == "expense")
        ]
        
        # Group by category
        if not month_transactions.empty:
            category_spending = month_transactions.groupby("category")["amount"].sum().to_dict()
            current_spending.update(category_spending)
    
    # Prepare data for display
    for category, goal in budget_goals[current_month].items():
        amount = goal["amount"]
        currency = goal["currency"]
        spent = current_spending.get(category, 0)
        remaining = amount - spent
        
        budget_data.append({
            "Category": category,
            "Monthly Budget": f"{currency} {amount:.2f}",
            "Current Spending": f"{currency} {spent:.2f}",
            "Remaining": f"{currency} {remaining:.2f}",
            "raw_budget": amount,
            "raw_spent": spent,
            "raw_remaining": remaining,
            "currency": currency
        })
    
    # Convert to dataframe
    if budget_data:
        budget_df = pd.DataFrame(budget_data)
        
        # Display table
        st.dataframe(budget_df[["Category", "Monthly Budget", "Current Spending", "Remaining"]], use_container_width=True)
        
        # Progress visualization
        st.markdown("### Budget Progress")
        
        for _, row in budget_df.iterrows():
            category = row["Category"]
            budget = row["raw_budget"]
            spent = row["raw_spent"]
            percentage = min(100, int((spent / budget) * 100)) if budget > 0 else 0
            
            # Progress bar color
            if percentage < 70:
                progress_color = "green"
            elif percentage < 90:
                progress_color = "orange"
            else:
                progress_color = "red"
            
            st.markdown(f"**{category}**")
            
            # Custom progress bar with color
            st.markdown(
                f"""
                <div style="border-radius:10px; border:1px solid #ccc; width:100%;">
                    <div style="background-color:{progress_color}; width:{percentage}%; 
                         border-radius:10px; padding:5px 0px; text-align:center; color:white;">
                        {percentage}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.text(f"{row['Current Spending']} of {row['Monthly Budget']} ({percentage}%)")
            st.markdown("---")
    else:
        st.info("No budget goals found for the current month.")
else:
    st.info("You haven't set any budget goals for the current month. Create one using the form above.")

# Add a section for budget recommendations
if not transactions.empty:
    st.markdown("### Budget Recommendations")
    
    # Filter for expense transactions
    expense_transactions = transactions[transactions["type"] == "expense"]
    
    if not expense_transactions.empty:
        # Get top spending categories
        top_spending = expense_transactions.groupby("category")["amount"].sum().sort_values(descending=True).head(3)
        
        # Generate recommendations
        recommendations = []
        
        for category, amount in top_spending.items():
            # Check if this category has a budget
            has_budget = current_month in budget_goals and category in budget_goals[current_month]
            
            if not has_budget:
                recommendations.append(f"Consider creating a budget for **{category}** as it's one of your top spending categories.")
            else:
                budget_amount = budget_goals[current_month][category]["amount"]
                if amount > budget_amount:
                    recommendations.append(f"You're over budget in **{category}**. Consider increasing your budget or reducing spending.")
        
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("Great job! You're on track with your budgets.")
    else:
        st.info("Add some expense transactions to get budget recommendations.")
