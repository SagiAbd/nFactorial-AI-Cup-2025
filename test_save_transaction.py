#!/usr/bin/env python3
"""
Test script to verify transaction saving functionality
"""
import os
import sys
from datetime import datetime

# Add parent directory to path to enable imports
sys.path.append(os.path.abspath('.'))

from core.data_manager import save_transaction

def main():
    """Test saving a transaction"""
    print("Testing transaction saving...")
    
    # Create a test transaction
    test_transaction = {
        'transaction_date': datetime.now(),
        'amount': 100.0,
        'category': 'Test',
        'currency': 'KZT',
        'transaction_type': 'income',
        'description': 'Test transaction'
    }
    
    print(f"Test transaction data: {test_transaction}")
    
    # Save the transaction
    result = save_transaction(**test_transaction)
    
    print(f"Save result: {result}")
    
    # Try with a string date to see if it fails
    try:
        print("\nTesting with string date (should fail):")
        test_transaction2 = {
            'date': '2025-06-01',  # Using 'date' instead of 'transaction_date'
            'amount': 200.0,
            'category': 'Test',
            'currency': 'KZT',
            'transaction_type': 'expense',
            'description': 'Test transaction 2'
        }
        
        print(f"Test transaction data: {test_transaction2}")
        
        # This should fail
        result = save_transaction(**test_transaction2)
        print(f"Save result: {result}")
    except Exception as e:
        print(f"Error as expected: {e}")
    
    # Check if the file exists and has content
    transactions_file = os.path.join("data", "transactions.csv")
    if os.path.exists(transactions_file):
        file_size = os.path.getsize(transactions_file)
        print(f"\nTransactions file size: {file_size} bytes")
        
        if file_size > 0:
            print("File has content")
            
            # Read the file
            try:
                with open(transactions_file, 'r') as f:
                    content = f.read()
                    print(f"File content: {content}")
            except Exception as e:
                print(f"Error reading file: {e}")
        else:
            print("File is empty")
    else:
        print("Transactions file does not exist")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 