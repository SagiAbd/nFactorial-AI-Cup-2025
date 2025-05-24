#!/usr/bin/env python3
"""
Debug script to test transaction saving functionality
"""
import os
import pandas as pd
from datetime import datetime
from core.data_manager import save_transaction, load_transactions, TRANSACTIONS_FILE

def main():
    print("\n--- Transaction Save Debug Test ---\n")
    
    # Check if data directory and transactions file exist
    data_dir = "data"
    print(f"Checking data directory: {os.path.abspath(data_dir)}")
    if os.path.exists(data_dir):
        print(f"Data directory exists: {data_dir}")
    else:
        print(f"Data directory doesn't exist: {data_dir}")
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")
    
    print(f"Transactions file path: {os.path.abspath(TRANSACTIONS_FILE)}")
    if os.path.exists(TRANSACTIONS_FILE):
        print(f"Transactions file exists")
        # Load existing transactions
        try:
            df = pd.read_csv(TRANSACTIONS_FILE)
            print(f"Loaded {len(df)} existing transactions")
        except Exception as e:
            print(f"Error loading transactions: {e}")
    else:
        print(f"Transactions file doesn't exist")
    
    # Create a test transaction
    test_transaction = {
        'transaction_date': datetime.now(),
        'amount': 1000.0,
        'category': 'Test',
        'currency': 'KZT',
        'transaction_type': 'income',
        'description': 'Debug test transaction'
    }
    
    print("\nAttempting to save test transaction:")
    print(test_transaction)
    
    # Try to save the transaction
    result = save_transaction(**test_transaction)
    
    if result:
        print("\n✅ Transaction saved successfully!")
        
        # Verify by loading again
        try:
            df = pd.read_csv(TRANSACTIONS_FILE)
            print(f"Transactions file now has {len(df)} transactions")
            print("\nLast 5 transactions:")
            print(df.tail(5))
        except Exception as e:
            print(f"Error verifying save: {e}")
    else:
        print("\n❌ Failed to save transaction")
    
    print("\n--- End of Test ---\n")

if __name__ == "__main__":
    main() 