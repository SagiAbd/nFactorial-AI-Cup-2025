#!/usr/bin/env python3
"""
Test script for the improved transaction parser.
This script tests the improved parsing capabilities on a sample file.
"""
import os
import sys
import argparse
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import transaction processor
from utils.transaction_processor import TransactionProcessor

def main():
    """Main function to test the transaction parser"""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test the transaction parser')
    parser.add_argument('file', help='Path to the file to test parsing on')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Check if file exists
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return 1
    
    # Initialize transaction processor
    try:
        logger.info("Initializing transaction processor")
        processor = TransactionProcessor()
    except Exception as e:
        logger.error(f"Error initializing transaction processor: {e}")
        return 1
    
    # Process the file
    try:
        logger.info(f"Processing file: {args.file}")
        result = processor.process_file(args.file)
        
        if not result["success"]:
            logger.error(f"Error processing file: {result.get('error', 'Unknown error')}")
            return 1
        
        transactions = result["transactions"]
        
        if not transactions:
            logger.warning("No transactions found in file")
            return 0
        
        # Print the extracted transactions
        logger.info(f"Successfully extracted {len(transactions)} transactions:")
        for i, tx in enumerate(transactions):
            logger.info(f"Transaction {i+1}:")
            for key, value in tx.items():
                logger.info(f"  {key}: {value}")
            logger.info("-" * 40)
        
        logger.info("Parser test completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 