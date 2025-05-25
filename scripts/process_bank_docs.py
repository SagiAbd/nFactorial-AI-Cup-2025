#!/usr/bin/env python3
"""
Command-line script to process bank statements and receipts
"""
import os
import sys
import argparse
import pandas as pd
from datetime import datetime

# Add parent directory to path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.transaction_processor import TransactionProcessor

def main():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Process bank statements and receipts')
    
    parser.add_argument('file', nargs='+', help='Files to process (CSV, PDF, XLSX, PNG, JPG)')
    parser.add_argument('--output', '-o', default='data/processed_transactions.csv', 
                        help='Output CSV file (default: data/processed_transactions.csv)')
    parser.add_argument('--config', '-c', default='data/config.json',
                        help='Configuration file (default: data/config.json)')
    parser.add_argument('--append', '-a', action='store_true',
                        help='Append to existing output file instead of overwriting')
    parser.add_argument('--api-key', help='OpenAI API key (optional)')
    
    args = parser.parse_args()
    
    # Validate files
    valid_files = []
    for file_path in args.file:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue
            
        file_ext = file_path.split('.')[-1].lower()
        if file_ext not in ['csv', 'pdf', 'xlsx', 'xls', 'jpg', 'jpeg', 'png']:
            print(f"Warning: Unsupported file format: {file_path}")
            continue
            
        valid_files.append(file_path)
    
    if not valid_files:
        print("Error: No valid files to process")
        return 1
    
    # Create processor
    try:
        processor = TransactionProcessor(api_key=args.api_key, config_path=args.config)
    except Exception as e:
        print(f"Error initializing processor: {e}")
        return 1
    
    # Process files
    all_transactions = []
    
    for file_path in valid_files:
        print(f"Processing {file_path}...")
        
        transactions = processor.process_file(file_path)
        
        if transactions.empty:
            print(f"  No transactions found in {file_path}")
            continue
        
        print(f"  Found {len(transactions)} transactions")
        all_transactions.append(transactions)
    
    if not all_transactions:
        print("No transactions found in any files")
        return 0
    
    # Combine all transactions
    combined_df = pd.concat(all_transactions, ignore_index=True)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save to CSV
    if args.append and os.path.exists(args.output):
        # Read existing file and append
        existing_df = pd.read_csv(args.output)
        combined_df = pd.concat([existing_df, combined_df], ignore_index=True)
    
    # Save to output file
    combined_df.to_csv(args.output, index=False)
    print(f"Saved {len(combined_df)} transactions to {args.output}")
    
    # Print summary
    print("\nTransaction summary:")
    print(f"  Total transactions: {len(combined_df)}")
    
    # Count unclear items
    unclear_count = combined_df['description'].str.contains('[UNCLEAR]', case=False).sum()
    unclear_count += combined_df['date'].str.contains('[UNCLEAR]', case=False).sum()
    unclear_count += combined_df['amount'].astype(str).str.contains('[UNCLEAR]', case=False).sum()
    
    if unclear_count > 0:
        print(f"  Transactions with unclear data: {unclear_count}")
    
    # Count by type
    type_counts = combined_df['type'].value_counts()
    for type_name, count in type_counts.items():
        print(f"  {type_name.capitalize()}: {count}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 