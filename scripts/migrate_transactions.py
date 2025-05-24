"""
Script to migrate existing transactions to the new column order format.
"""
import os
import pandas as pd
import sys
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Configuration
DATA_DIR = "data"
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.csv")

def categorize_transaction(description, amount):
    """Simple function to categorize transactions based on description and amount"""
    # Income categorization
    if amount > 0:
        if any(keyword in description.lower() for keyword in ["salary", "wage", "payment", "deposit"]):
            return "Salary"
        elif any(keyword in description.lower() for keyword in ["bonus", "award"]):
            return "Bonus"
        elif any(keyword in description.lower() for keyword in ["dividend", "interest", "return"]):
            return "Investment"
        elif any(keyword in description.lower() for keyword in ["refund", "return", "cashback"]):
            return "Refund"
        elif any(keyword in description.lower() for keyword in ["gift", "present"]):
            return "Gift"
        else:
            return "Other Income"
    
    # Expense categorization
    else:
        if any(keyword in description.lower() for keyword in ["grocery", "food", "restaurant", "cafe", "dinner", "lunch"]):
            return "Food"
        elif any(keyword in description.lower() for keyword in ["taxi", "uber", "transport", "bus", "train", "fuel", "gas"]):
            return "Transport"
        elif any(keyword in description.lower() for keyword in ["electricity", "water", "gas", "internet", "phone", "bill"]):
            return "Utilities"
        elif any(keyword in description.lower() for keyword in ["shopping", "amazon", "purchase"]):
            return "Shopping"
        elif any(keyword in description.lower() for keyword in ["cinema", "movie", "entertainment", "game"]):
            return "Entertainment"
        elif any(keyword in description.lower() for keyword in ["health", "doctor", "medicine", "hospital"]):
            return "Health"
        elif any(keyword in description.lower() for keyword in ["education", "course", "book", "learn"]):
            return "Education"
        elif any(keyword in description.lower() for keyword in ["rent", "mortgage", "home"]):
            return "Housing"
        else:
            return "Other Expense"

def migrate_transactions():
    """Migrate existing transactions to the new column order format."""
    if not os.path.exists(TRANSACTIONS_FILE):
        print(f"Transactions file not found: {TRANSACTIONS_FILE}")
        return
    
    print(f"Migrating transactions in {TRANSACTIONS_FILE}...")
    
    try:
        # Read the existing transactions
        df = pd.read_csv(TRANSACTIONS_FILE)
        
        # Keep a backup of the original file
        backup_file = f"{TRANSACTIONS_FILE}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}"
        df.to_csv(backup_file, index=False)
        print(f"Backup created: {backup_file}")
        
        # Ensure all required columns exist
        for col in ['date', 'description', 'amount', 'type', 'category', 'currency']:
            if col not in df.columns:
                if col == 'type':
                    df[col] = df.apply(lambda row: 'income' if row['amount'] > 0 else 'expense', axis=1)
                elif col == 'description' and col not in df.columns:
                    df[col] = ''
                elif col == 'category':
                    # If category doesn't exist, create it
                    if 'type' in df.columns:
                        # For income, use a more specific category than just 'Income'
                        df[col] = df.apply(
                            lambda row: categorize_transaction(row.get('description', ''), row['amount']) 
                            if pd.isna(row.get('category', None)) else row.get('category'),
                            axis=1
                        )
                    else:
                        df[col] = 'Uncategorized'
                else:
                    df[col] = ''
        
        # Reorder columns
        df = df[['date', 'description', 'amount', 'type', 'category', 'currency']]
        
        # Save with the new format
        df.to_csv(TRANSACTIONS_FILE, index=False)
        print(f"Migration completed successfully. {len(df)} transactions updated.")
        
    except Exception as e:
        print(f"Error during migration: {str(e)}")

if __name__ == "__main__":
    migrate_transactions() 