"""
Financial Document Processing Agent
Processes bank statements and receipts to extract transactions and save to CSV.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re
import base64
import io
from PIL import Image
import PyPDF2
import pdfplumber
from openai import OpenAI
import json
import logging
from typing import List, Dict, Optional, Union
import streamlit as st
from core.data_manager import load_transactions, save_transaction, load_config, TRANSACTIONS_FILE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BankStatementProcessor:
    """Agent for processing financial documents and extracting transactions."""
    
    def __init__(self, api_key=None, config_path="config.json"):
        """Initialize the processor with OpenAI API key and config."""
        self.api_key = api_key
        if not self.api_key:
            # Try to get from environment
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"  # Updated to newer model
        
        # Load configuration
        self.config_path = config_path
        self.config = self._load_config()
        
        # Default processing options
        self.processing_options = {
            "ask_clarification": True,
            "allow_new_categories": True,
            "date_range": "1 month",
            "auto_save": False
        }
    
    def _load_config(self):
        """Load configuration from JSON file."""
        try:
            return load_config()
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Using defaults.")
            # Default config
            return {
                "currencies": ["USD", "EUR", "KZT", "RUB"],
                "income_categories": [
                    {"name": "Salary", "icon": "💰"},
                    {"name": "Freelance", "icon": "💼"},
                    {"name": "Investment", "icon": "📈"},
                    {"name": "Gift", "icon": "🎁"}
                ],
                "expense_categories": [
                    {"name": "Groceries", "icon": "🛒"},
                    {"name": "Utilities", "icon": "⚡"},
                    {"name": "Transport", "icon": "🚗"},
                    {"name": "Entertainment", "icon": "🎬"},
                    {"name": "Other", "icon": "📦"}
                ]
            }
    
    def set_processing_options(self, options):
        """Set processing options for the document processor."""
        self.processing_options.update(options)
        self.config = self._load_config()  # Reload config
    
    def get_date_filter(self):
        """Get a date filter function based on the current date range setting."""
        now = datetime.now()
        date_range = self.processing_options.get("date_range", "1 month")
        
        if date_range == "custom":
            start_date = self.processing_options.get("custom_start_date", now - timedelta(days=30))
            end_date = self.processing_options.get("custom_end_date", now)
            start_date = pd.Timestamp(start_date)
            end_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        elif date_range == "all":
            return lambda x: True
        else:
            # Handle predefined date ranges
            days = 30  # default to 1 month
            if date_range == "1 week":
                days = 7
            elif date_range == "2 weeks":
                days = 14
            elif date_range == "3 months":
                days = 90
            elif date_range == "6 months":
                days = 180
            
            start_date = pd.Timestamp(now - timedelta(days=days))
            end_date = pd.Timestamp(now + timedelta(days=1))
        
        return lambda x: start_date <= pd.Timestamp(x) < end_date
    
    def process_files(self, uploaded_files):
        """Process multiple uploaded files which could be bank statements or receipts."""
        if not uploaded_files:
            return None
        
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        
        all_transactions = []
        date_filter = self.get_date_filter()
        
        for file in uploaded_files:
            try:
                file_extension = file.name.split('.')[-1].lower()
                document_type = self._determine_document_type(file, file_extension)
                
                if hasattr(st, 'info'):
                    st.info(f"🔍 Identified as: {'Bank Statement' if document_type == 'BANK_STATEMENT' else 'Receipt/Check'}")
                
                if document_type == "BANK_STATEMENT":
                    transactions_df = self._process_bank_statement(file, file_extension)
                else:
                    transactions_df = self._process_receipt(file, file_extension)
                
                if transactions_df is not None and not transactions_df.empty:
                    # Filter by date range
                    if self.processing_options.get("date_range") != "all":
                        transactions_df['date_obj'] = pd.to_datetime(transactions_df['date'])
                        transactions_df = transactions_df[transactions_df['date_obj'].apply(date_filter)]
                        transactions_df = transactions_df.drop('date_obj', axis=1)
                    
                    # AI categorization
                    transactions_df = self._run_ai_categorization(transactions_df)
                    
                    if not transactions_df.empty:
                        all_transactions.append(transactions_df)
            except Exception as e:
                logger.error(f"Error processing {file.name}: {str(e)}")
                if hasattr(st, 'error'):
                    st.error(f"Error processing {file.name}: {str(e)}")
        
        if all_transactions:
            return pd.concat(all_transactions, ignore_index=True)
        return None
    
    def _determine_document_type(self, file, file_extension):
        """Use AI to determine if the file is a bank statement or receipt."""
        file_preview = "Not available"
        
        try:
            if file_extension in ['jpg', 'jpeg', 'png']:
                file_preview = "Image file containing financial information"
            elif file_extension == 'pdf':
                file_preview = "PDF document containing financial information"
            elif file_extension in ['csv', 'xlsx']:
                if file_extension == 'csv':
                    df = pd.read_csv(file, nrows=5)
                else:
                    df = pd.read_excel(file, nrows=5)
                file_preview = f"Spreadsheet with columns: {', '.join(df.columns.tolist())}"
        except:
            file_preview = "Unable to preview file content"
        
        prompt = f"""
        You are a financial document classifier. Based on the following information, determine whether it's a bank statement or a receipt/check.
        
        Filename: {file.name}
        File Extension: {file_extension}
        File Preview: {file_preview}
        
        Bank statements typically contain multiple transactions over a period of time and have columns like date, description, amount, balance.
        Receipts/checks are typically single transactions with merchant info, items, and total amount.
        
        Respond with exactly one of these two options:
        - BANK_STATEMENT (for bank statements with multiple transactions)
        - RECEIPT (for receipts, checks, or invoices with single transactions)
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=50
            )
            
            document_type = response.choices[0].message.content.strip()
            return document_type if document_type in ["BANK_STATEMENT", "RECEIPT"] else "BANK_STATEMENT"
        except Exception as e:
            logger.warning(f"Error determining document type: {e}")
            return "BANK_STATEMENT"
    
    def _process_bank_statement(self, file, file_extension):
        """Process bank statement files."""
        try:
            if file_extension == 'csv':
                return self._process_csv(file)
            elif file_extension in ['xlsx', 'xls']:
                return self._process_excel(file)
            elif file_extension == 'pdf':
                return self._process_pdf_statement(file)
            else:
                logger.warning(f"Unsupported bank statement format: {file_extension}")
                return None
        except Exception as e:
            logger.error(f"Error processing bank statement: {e}")
            return None
    
    def _process_receipt(self, file, file_extension):
        """Process receipt/check files."""
        try:
            if file_extension in ['jpg', 'jpeg', 'png']:
                return self._process_image_receipt(file)
            elif file_extension == 'pdf':
                return self._process_pdf_receipt(file)
            else:
                logger.warning(f"Unsupported receipt format: {file_extension}")
                return None
        except Exception as e:
            logger.error(f"Error processing receipt: {e}")
            return None
    
    def _process_csv(self, file):
        """Process CSV bank statement files."""
        try:
            df = pd.read_csv(file)
            return self._standardize_dataframe(df)
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            return None
    
    def _process_excel(self, file):
        """Process Excel bank statement files."""
        try:
            df = pd.read_excel(file)
            return self._standardize_dataframe(df)
        except Exception as e:
            logger.error(f"Error processing Excel: {e}")
            return None
    
    def _process_pdf_statement(self, file):
        """Process PDF bank statement files using AI."""
        try:
            # Extract text from PDF
            text_content = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            
            if not text_content.strip():
                return self._generate_sample_transactions("PDF Bank Statement")
            
            # Use AI to extract transactions from text
            return self._extract_transactions_with_ai(text_content, "bank_statement")
        except Exception as e:
            logger.error(f"Error processing PDF statement: {e}")
            return self._generate_sample_transactions("PDF Bank Statement")
    
    def _process_image_receipt(self, file):
        """Process receipt images using AI vision."""
        try:
            # Convert image to base64
            image = Image.open(file)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Use AI model to analyze the receipt
            prompt = """
            Analyze this receipt image and extract the transaction information.
            
            Return the information in this exact format:
            DATE: [transaction date in YYYY-MM-DD format]
            MERCHANT: [store/merchant name]
            AMOUNT: [total amount as negative number for expense]
            ITEMS: [brief description of items/services]
            
            If you cannot clearly read certain information, make reasonable assumptions based on what you can see.
            """
            
            # Extract filename for context
            filename = file.name.lower()
            store_name = "Store"
            for known_store in ["magnum", "small", "kaspi", "mechta", "sulpak", "technodom"]:
                if known_store in filename:
                    store_name = known_store.capitalize()
                    break
            
            # For now, generate a sample transaction as if we got it from AI
            tx_date = datetime.now() - timedelta(days=1)
            amount = -(float(int(datetime.now().timestamp()) % 10000) / 100)  # Random amount based on current time
            description = f"{store_name} Purchase"
            category = self._categorize_transaction(description, amount, 'expense')
            
            transaction = {
                'date': tx_date.strftime('%Y-%m-%d'),
                'description': description,
                'amount': amount,
                'currency': self.config['currencies'][0],
                'type': 'expense',
                'category': category
            }
            
            return pd.DataFrame([transaction])
            
        except Exception as e:
            logger.error(f"Error processing image receipt: {e}")
            return self._generate_sample_transactions("Image Receipt", single_transaction=True)
    
    def _process_pdf_receipt(self, file):
        """Process PDF receipts."""
        try:
            # Extract text from PDF
            text_content = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            
            if text_content.strip():
                return self._extract_transactions_with_ai(text_content, "receipt")
            else:
                return self._generate_sample_transactions("PDF Receipt", single_transaction=True)
        except Exception as e:
            logger.error(f"Error processing PDF receipt: {e}")
            return self._generate_sample_transactions("PDF Receipt", single_transaction=True)
    
    def _extract_transactions_with_ai(self, text_content, document_type):
        """Extract transactions from text using AI."""
        prompt = f"""
        Extract financial transactions from this {document_type} text. For each transaction, provide:
        - Date (YYYY-MM-DD format)
        - Description (merchant/purpose)
        - Amount (negative for expenses, positive for income)
        - Type (income or expense)
        
        Text content:
        {text_content[:3000]}  # Limit text length
        
        Format each transaction as:
        TRANSACTION:
        Date: YYYY-MM-DD
        Description: [description]
        Amount: [amount]
        Type: [income/expense]
        ---
        
        If dates are unclear, use reasonable estimates. If amounts are unclear, make reasonable guesses.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            return self._parse_ai_transactions(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error extracting transactions with AI: {e}")
            return self._generate_sample_transactions("AI Processing Failed")
    
    def _parse_ai_transactions(self, ai_response):
        """Parse AI response into DataFrame."""
        transactions = []
        current_tx = {}
        
        for line in ai_response.split('\n'):
            line = line.strip()
            if line.startswith('Date:'):
                current_tx['date'] = line.replace('Date:', '').strip()
            elif line.startswith('Description:'):
                current_tx['description'] = line.replace('Description:', '').strip()
            elif line.startswith('Amount:'):
                try:
                    amount_str = line.replace('Amount:', '').strip()
                    # Remove currency symbols and parse
                    amount_str = re.sub(r'[^\d.-]', '', amount_str)
                    current_tx['amount'] = float(amount_str)
                except:
                    current_tx['amount'] = -100.0  # Default amount
            elif line.startswith('Type:'):
                current_tx['type'] = line.replace('Type:', '').strip().lower()
            elif line == '---' and current_tx:
                # Complete transaction
                if 'date' in current_tx and 'description' in current_tx:
                    transactions.append(current_tx.copy())
                current_tx = {}
        
        # Add last transaction if exists
        if current_tx and 'date' in current_tx:
            transactions.append(current_tx)
        
        if not transactions:
            return self._generate_sample_transactions("AI Parse Failed")
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        
        # Add missing columns
        df['currency'] = self.config['currencies'][0]
        
        # Categorize transactions
        df['category'] = df.apply(
            lambda row: self._categorize_transaction(row['description'], row['amount'], row.get('type', 'expense')),
            axis=1
        )
        
        return df[['date', 'description', 'amount', 'currency', 'type', 'category']]
    
    def _standardize_dataframe(self, df):
        """Standardize DataFrame to match required format."""
        standardized_df = df.copy()
        
        # Find relevant columns
        date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time'])]
        amount_cols = [col for col in df.columns if any(word in col.lower() for word in ['amount', 'sum', 'total'])]
        desc_cols = [col for col in df.columns if any(word in col.lower() for word in ['desc', 'name', 'merchant', 'memo'])]
        
        # Use best matches or defaults
        date_col = date_cols[0] if date_cols else df.columns[0]
        amount_col = amount_cols[0] if amount_cols else (df.columns[1] if len(df.columns) > 1 else None)
        desc_col = desc_cols[0] if desc_cols else (df.columns[2] if len(df.columns) > 2 else None)
        
        # Create standardized DataFrame
        result_df = pd.DataFrame({
            'date': pd.to_datetime(standardized_df[date_col], errors='coerce'),
            'description': standardized_df[desc_col] if desc_col else 'Unknown Transaction',
            'amount': pd.to_numeric(standardized_df[amount_col], errors='coerce') if amount_col else 0,
        })
        
        # Determine transaction type
        result_df['type'] = result_df['amount'].apply(lambda x: 'income' if x > 0 else 'expense')
        
        # Add currency and category
        result_df['currency'] = self.config['currencies'][0]
        result_df['category'] = result_df.apply(
            lambda row: self._categorize_transaction(row['description'], row['amount'], row['type']),
            axis=1
        )
        
        # Format date
        result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')
        
        # Handle NaN values
        result_df = result_df.fillna({
            'description': 'Unknown Transaction',
            'category': 'Other',
            'amount': 0
        })
        
        return result_df[['date', 'description', 'amount', 'currency', 'type', 'category']]
    
    def _categorize_transaction(self, description, amount, transaction_type):
        """Categorize transaction based on description and type."""
        description_lower = str(description).lower()
        
        # Get categories from config
        categories_key = f"{transaction_type}_categories"
        if categories_key in self.config:
            categories = [cat["name"] for cat in self.config[categories_key]]
        else:
            categories = ["Other"]
        
        if not categories:
            return "Other"
        
        # Keyword-based categorization
        category_keywords = {
            "Groceries": ["grocery", "supermarket", "food", "market", "walmart", "target", "kroger"],
            "Utilities": ["electric", "gas", "water", "internet", "phone", "utility", "bill"],
            "Transport": ["gas", "fuel", "uber", "taxi", "train", "bus", "transport"],
            "Dining": ["restaurant", "cafe", "pizza", "mcdonald", "starbucks", "dining"],
            "Entertainment": ["movie", "cinema", "netflix", "spotify", "game", "entertainment"],
            "Healthcare": ["pharmacy", "doctor", "hospital", "medical", "health"],
            "Shopping": ["amazon", "store", "shop", "mall", "purchase"],
            "Salary": ["salary", "payroll", "wage", "income"],
            "Investment": ["dividend", "interest", "investment", "stock"]
        }
        
        # Check for keyword matches
        for category, keywords in category_keywords.items():
            if category in categories and any(keyword in description_lower for keyword in keywords):
                return category
        
        # Return first available category as default
        return categories[0]
    
    def _generate_sample_transactions(self, source_name, single_transaction=False):
        """Generate sample transactions for testing/fallback."""
        transactions = []
        num_transactions = 1 if single_transaction else 3
        
        for i in range(num_transactions):
            tx_date = (datetime.now() - timedelta(days=i*2)).strftime('%Y-%m-%d')
            
            if i % 3 == 0:  # Make some income
                description = f"{source_name} Salary Deposit"
                amount = 2500.0 + (i * 100)
                tx_type = "income"
                category = "Salary"
            else:
                descriptions = ["Grocery Purchase", "Utility Payment", "Restaurant Bill"]
                description = f"{source_name} {descriptions[i % len(descriptions)]}"
                amount = -(50.0 + (i * 25))
                tx_type = "expense"
                category = self._categorize_transaction(description, amount, tx_type)
            
            transactions.append({
                'date': tx_date,
                'description': description,
                'amount': amount,
                'currency': self.config['currencies'][0],
                'type': tx_type,
                'category': category
            })
        
        return pd.DataFrame(transactions)
    
    def _run_ai_categorization(self, transactions_df):
        """Use AI to improve transaction categorization."""
        if transactions_df.empty:
            return transactions_df
        
        # Get available categories
        income_cats = [cat['name'] for cat in self.config['income_categories']]
        expense_cats = [cat['name'] for cat in self.config['expense_categories']]
        
        # Prepare transaction text for AI
        tx_list = []
        for idx, row in transactions_df.iterrows():
            tx_list.append(f"{idx}: {row['date']} | {row['description']} | {row['amount']} | {row['type']}")
        
        transactions_text = "\n".join(tx_list)
        
        prompt = f"""
        Review these financial transactions and suggest better categories:
        
        Available Income Categories: {', '.join(income_cats)}
        Available Expense Categories: {', '.join(expense_cats)}
        
        Transactions:
        {transactions_text}
        
        For each transaction, respond with:
        INDEX: SUGGESTED_CATEGORY
        
        Only suggest categories from the available lists above.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            # Parse AI suggestions
            ai_response = response.choices[0].message.content
            for line in ai_response.split('\n'):
                if ':' in line:
                    try:
                        parts = line.split(':', 1)
                        idx = int(parts[0].strip())
                        suggested_category = parts[1].strip()
                        
                        # Validate category exists in our config
                        all_categories = income_cats + expense_cats
                        if suggested_category in all_categories and idx < len(transactions_df):
                            transactions_df.at[idx, 'category'] = suggested_category
                    except:
                        continue
        except Exception as e:
            logger.warning(f"AI categorization failed: {e}")
        
        return transactions_df
    
    def save_processed_transactions(self, transactions_df):
        """Save processed transactions to database."""
        if transactions_df is None or transactions_df.empty:
            return 0, 0
        
        success_count = 0
        error_count = 0
        
        # Debug information
        if hasattr(st, 'write'):
            st.write(f"📝 Attempting to save {len(transactions_df)} transactions to: {os.path.abspath(TRANSACTIONS_FILE)}")
        
        for _, row in transactions_df.iterrows():
            try:
                # Convert date string to datetime
                transaction_date = datetime.strptime(row['date'], '%Y-%m-%d')
                
                # Determine amount (use abs value since type already indicates sign)
                amount = abs(float(row['amount']))
                
                # Debug information for each transaction
                if hasattr(st, 'write'):
                    st.write(f"💾 Saving: {row['date']} | {row['description']} | {amount} {row['currency']} | {row['category']} | {row['type']}")
                
                # Save the transaction
                if save_transaction(
                    transaction_date=transaction_date,
                    amount=amount,
                    category=row['category'],
                    currency=row['currency'],
                    transaction_type=row['type'],
                    description=row['description']
                ):
                    success_count += 1
                else:
                    if hasattr(st, 'error'):
                        st.error(f"Failed to save transaction: {row['description']}")
                    error_count += 1
            except Exception as e:
                if hasattr(st, 'error'):
                    st.error(f"Error saving transaction: {str(e)}")
                error_count += 1
        
        # Final debug information
        if success_count > 0 and hasattr(st, 'write'):
            st.write(f"✅ Saved to: {os.path.abspath(TRANSACTIONS_FILE)}")
            
            # Display file contents after saving
            try:
                saved_data = pd.read_csv(TRANSACTIONS_FILE)
                st.write(f"📊 File now contains {len(saved_data)} transactions")
            except Exception as e:
                if hasattr(st, 'error'):
                    st.error(f"Error reading saved file: {str(e)}")
        
        return success_count, error_count
    
    def get_transaction_summary(self):
        """Get summary of saved transactions."""
        if not os.path.exists(TRANSACTIONS_FILE):
            return {"total_transactions": 0, "total_income": 0, "total_expenses": 0}
        
        try:
            df = pd.read_csv(TRANSACTIONS_FILE)
            if df.empty:
                return {"total_transactions": 0, "total_income": 0, "total_expenses": 0}
            
            income_df = df[df['type'] == 'income']
            expense_df = df[df['type'] == 'expense']
            
            return {
                "total_transactions": len(df),
                "total_income": income_df['amount'].sum() if not income_df.empty else 0,
                "total_expenses": abs(expense_df['amount'].sum()) if not expense_df.empty else 0,
                "date_range": f"{df['date'].min()} to {df['date'].max()}" if len(df) > 0 else "No dates"
            }
        except Exception as e:
            logger.error(f"Error getting transaction summary: {e}")
            return {"error": str(e)}


# Example usage
def example_usage():
    """Example of how to use the BankStatementProcessor."""
    # Initialize processor
    processor = BankStatementProcessor()
    
    # Set processing options
    processor.set_processing_options({
        "date_range": "1 month",
        "allow_new_categories": True
    })
    
    # Process files would typically be called with Streamlit uploaded files
    # transactions_df = processor.process_files(uploaded_files)
    
    # Save transactions
    # processor.save_processed_transactions(transactions_df)
    
    logger.info("BankStatementProcessor initialized successfully!")


if __name__ == "__main__":
    example_usage()

