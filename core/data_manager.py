import streamlit as st
import pandas as pd
import json
import os
from datetime import date, timedelta
from dotenv import load_dotenv


# Configuration
DATA_DIR = "data"
CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.csv")
GOALS_FILE = os.path.join(DATA_DIR, "goals.json")

# Default configurations
DEFAULT_CONFIG = {
    "currencies": ["KZT", "USD", "EUR"],
    "expense_categories": [
        {"name": "Food", "icon": "🍔"},
        {"name": "Transport", "icon": "🚗"},
        {"name": "Utilities", "icon": "💡"},
        {"name": "Shopping", "icon": "🛍️"},
        {"name": "Entertainment", "icon": "🎮"},
        {"name": "Health", "icon": "🏥"},
        {"name": "Education", "icon": "📚"},
        {"name": "Housing", "icon": "🏠"}
    ],
    "income_categories": [
        {"name": "Salary", "icon": "💼"},
        {"name": "Freelance", "icon": "💻"},
        {"name": "Investment", "icon": "📈"},
        {"name": "Gift", "icon": "🎁"},
        {"name": "Bonus", "icon": "🎯"},
        {"name": "Refund", "icon": "💸"}
    ]
}

DEFAULT_GOALS = {
    "monthly_budget": 100000,  # KZT
    "savings_target": 50000,   # KZT per month
    "category_limits": {
        "Food": 30000,
        "Transport": 15000,
        "Entertainment": 10000,
        "Shopping": 20000
    }
}


def ensure_data_directory():
    """Ensure data directory exists"""
    os.makedirs(DATA_DIR, exist_ok=True)


def load_config():
    """Load configuration from JSON file"""
    ensure_data_directory()
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return DEFAULT_CONFIG


def save_config(config):
    """Save configuration to JSON file"""
    ensure_data_directory()
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving config: {e}")
        return False


def load_goals():
    """Load financial goals from JSON file"""
    ensure_data_directory()
    try:
        if os.path.exists(GOALS_FILE):
            with open(GOALS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            save_goals(DEFAULT_GOALS)
            return DEFAULT_GOALS
    except Exception as e:
        st.error(f"Error loading goals: {e}")
        return DEFAULT_GOALS


def save_goals(goals):
    """Save financial goals to JSON file"""
    ensure_data_directory()
    try:
        with open(GOALS_FILE, 'w', encoding='utf-8') as f:
            json.dump(goals, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving goals: {e}")
        return False


def initialize_session_state():
    """Initialize session state variables"""
    config = load_config()
    goals = load_goals()
    
    if "selected_type" not in st.session_state:
        st.session_state.selected_type = None
    if "selected_currency" not in st.session_state:
        st.session_state.selected_currency = None
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None
    if "form_submitted" not in st.session_state:
        st.session_state.form_submitted = False
    if "config" not in st.session_state:
        st.session_state.config = config
    if "goals" not in st.session_state:
        st.session_state.goals = goals
    if "messages" not in st.session_state:
        st.session_state.messages = []


def load_transactions():
    """Load transactions from CSV file"""
    ensure_data_directory()
    try:
        if os.path.exists(TRANSACTIONS_FILE):
            df = pd.read_csv(TRANSACTIONS_FILE)
            
            # Handle date column (now without time)
            try:
                # Convert date strings to datetime objects
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                
                # Check if 'datetime' column exists instead of 'date'
                if 'date' not in df.columns and 'datetime' in df.columns:
                    df['date'] = pd.to_datetime(df['datetime'], errors='coerce')
                    df = df.drop('datetime', axis=1)
            except Exception as e:
                st.warning(f"Warning when processing dates: {e}")
            
            # Drop any rows with invalid dates
            df = df.dropna(subset=['date'])
            
            # Check column order
            current_columns = list(df.columns)
            expected_columns = ['date', 'description', 'amount', 'currency', 'type', 'category']
            
            # Ensure all required columns exist
            for col in expected_columns:
                if col not in current_columns:
                    if col == 'type':
                        df[col] = df.apply(lambda row: 'income' if row['amount'] > 0 else 'expense', axis=1)
                    elif col == 'category':
                        df[col] = 'Uncategorized'
                    elif col == 'description':
                        df[col] = ''
                    elif col == 'currency':
                        df[col] = 'KZT'  # Default currency
            
            # Reorder columns if needed
            if current_columns != expected_columns:
                # Use only columns that exist in the DataFrame
                ordered_columns = [col for col in expected_columns if col in df.columns]
                df = df[ordered_columns]
                
                # Save with the new column order (if needed)
                # df.to_csv(TRANSACTIONS_FILE, index=False)
            
            return df
        else:
            # Create empty dataframe with the correct column order
            df = pd.DataFrame(columns=['date', 'description', 'amount', 'currency', 'type', 'category'])
            df.to_csv(TRANSACTIONS_FILE, index=False)
            return df
    except Exception as e:
        st.error(f"Error loading transactions: {e}")
        return pd.DataFrame(columns=['date', 'description', 'amount', 'currency', 'type', 'category'])


def save_transaction(transaction_date, amount, category, currency, transaction_type, description=""):
    """Save a new transaction to CSV file"""
    try:
        print(f"Debug: save_transaction called with {transaction_date}, {amount}, {category}, {currency}, {transaction_type}, {description}")
        
        # Ensure data directory exists
        ensure_data_directory()
        
        transactions = load_transactions()
        print(f"Debug: Loaded existing transactions, count: {len(transactions)}")
        
        # Format date as YYYY-MM-DD (without time)
        date_str = transaction_date.strftime('%Y-%m-%d')
        
        new_transaction = {
            'date': date_str,
            'description': description,
            'amount': amount if transaction_type == 'income' else -abs(amount),
            'currency': currency,
            'type': transaction_type,
            'category': category
        }
        print(f"Debug: New transaction data: {new_transaction}")
        
        new_df = pd.DataFrame([new_transaction])
        transactions = pd.concat([transactions, new_df], ignore_index=True)
        print(f"Debug: Concatenated transactions, new count: {len(transactions)}")
        
        # Save to file
        transactions.to_csv(TRANSACTIONS_FILE, index=False)
        print(f"Debug: Saved transactions to {TRANSACTIONS_FILE}")
        
        return True
    except Exception as e:
        print(f"Debug ERROR: Error saving transaction: {e}")
        if hasattr(st, 'error'):
            st.error(f"Error saving transaction: {e}")
        return False


def add_category(transaction_type, name, icon):
    """Add new category and save to config"""
    config = st.session_state.config
    categories_key = f"{transaction_type}_categories"
    
    existing_names = [cat['name'] for cat in config[categories_key]]
    if name not in existing_names:
        config[categories_key].append({"name": name, "icon": icon})
        st.session_state.config = config
        save_config(config)
        return True
    return False


def get_financial_summary():
    """Get comprehensive financial summary"""
    transactions = load_transactions()
    
    if transactions.empty:
        return {
            "total_income": 0,
            "total_expenses": 0,
            "balance": 0,
            "monthly_income": 0,
            "monthly_expenses": 0,
            "top_categories": [],
            "recent_trend": "No data"
        }
    
    # Current month data
    current_month = pd.Timestamp(date.today().replace(day=1))
    monthly_data = transactions[transactions['date'] >= current_month]
    
    # Calculate totals
    total_income = transactions[transactions['amount'] > 0]['amount'].sum()
    total_expenses = abs(transactions[transactions['amount'] < 0]['amount'].sum())
    balance = total_income - total_expenses
    
    monthly_income = monthly_data[monthly_data['amount'] > 0]['amount'].sum()
    monthly_expenses = abs(monthly_data[monthly_data['amount'] < 0]['amount'].sum())
    
    # Top spending categories
    expense_by_category = transactions[transactions['amount'] < 0].groupby('category')['amount'].sum().abs().sort_values(ascending=False)
    top_categories = expense_by_category.head(3).to_dict()
    
    # Recent trend (last 7 days vs previous 7 days)
    last_week = pd.Timestamp(date.today() - timedelta(days=7))
    prev_week = pd.Timestamp(date.today() - timedelta(days=14))
    
    recent_expenses = abs(transactions[
        (transactions['date'] >= last_week) & 
        (transactions['amount'] < 0)
    ]['amount'].sum())
    
    prev_expenses = abs(transactions[
        (transactions['date'] >= prev_week) & 
        (transactions['date'] < last_week) & 
        (transactions['amount'] < 0)
    ]['amount'].sum())
    
    if prev_expenses > 0:
        trend_pct = ((recent_expenses - prev_expenses) / prev_expenses) * 100
        if trend_pct > 10:
            recent_trend = f"Spending increased by {trend_pct:.1f}%"
        elif trend_pct < -10:
            recent_trend = f"Spending decreased by {abs(trend_pct):.1f}%"
        else:
            recent_trend = "Spending stable"
    else:
        recent_trend = "Not enough data"
    
    return {
        "total_income": total_income,
        "total_expenses": total_expenses,
        "balance": balance,
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "top_categories": top_categories,
        "recent_trend": recent_trend
    }


def get_category_spending(category, days=30):
    """Get spending for a specific category in the last X days"""
    transactions = load_transactions()
    
    if transactions.empty:
        return 0
    
    # Calculate cutoff date
    cutoff_date = pd.Timestamp(date.today() - timedelta(days=days))
    
    # Filter transactions
    category_spending = abs(transactions[
        (transactions['category'] == category) &
        (transactions['date'] >= cutoff_date)
    ]['amount'].sum())
    
    return category_spending


def get_monthly_comparison():
    """Get month-by-month spending and income for comparison"""
    transactions = load_transactions()
    
    if transactions.empty:
        return pd.DataFrame()
    
    # Create month periods for grouping
    transactions['month'] = transactions['date'].dt.to_period('M')
    
    # Group by month
    monthly_summary = transactions.groupby('month').agg({
        'amount': lambda x: {
            'income': x[x > 0].sum(),
            'expenses': abs(x[x < 0].sum()),
            'balance': x.sum()
        }
    }).to_dict()
    
    return monthly_summary

# Check if running on Streamlit Cloud or locally
if 'OPENAI_API_KEY' in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    # Local development fallback
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") 