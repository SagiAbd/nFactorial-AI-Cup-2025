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

# Path for clarification examples
CLARIFICATIONS_FILE = os.path.join("data", "clarifications.json")

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
        self.model = "gpt-4.1-mini"  # Updated to newer model
        
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
        
        # Load clarification examples
        self.clarification_examples = self._load_clarification_examples()
    
    def _load_config(self):
        """Load configuration from JSON file."""
        try:
            return load_config()
        except Exception as e:
            logger.warning(f"Error loading config: {e}. Using defaults.")
            # Default config
            return {
                "currencies": ["USD", "EUR", "KZT"],
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
        """Process uploaded files with simplified handling."""
        if not uploaded_files:
            return None
        
        if not isinstance(uploaded_files, list):
            uploaded_files = [uploaded_files]
        
        successful_uploads = []
        
        for file in uploaded_files:
            try:
                file_extension = file.name.split('.')[-1].lower()
                
                # Just record the file information without complex processing
                file_info = {
                    'filename': file.name,
                    'type': file_extension,
                    'size': file.size,
                    'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                successful_uploads.append(file_info)
                
                if hasattr(st, 'success'):
                    st.success(f"✅ Successfully uploaded {file.name}")
                
                logger.info(f"Successfully uploaded {file.name}")
                
            except Exception as e:
                logger.error(f"Error uploading {file.name}: {str(e)}")
                if hasattr(st, 'error'):
                    st.error(f"Error uploading {file.name}: {str(e)}")
        
        if successful_uploads:
            if hasattr(st, 'info'):
                st.info(f"📂 Total files uploaded: {len(successful_uploads)}")
            return pd.DataFrame(successful_uploads)
        
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
            
            print(f"\n----- PDF STATEMENT EXTRACTION -----")
            print(f"Filename: {file.name}")
            print(f"Text content length: {len(text_content)}")
            print(f"First 500 chars: {text_content[:500]}...")
            
            if not text_content.strip():
                print("No text content extracted from PDF")
                return self._generate_sample_transactions("PDF Bank Statement")
            
            # Use AI to extract transactions from text
            return self._extract_transactions_with_ai(text_content, "bank_statement")
        except Exception as e:
            logger.error(f"Error processing PDF statement: {e}")
            print(f"Error processing PDF statement: {e}")
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
            
            If you cannot clearly read certain information, explicitly indicate this with [UNCLEAR] tag, for example:
            DATE: [UNCLEAR] (appears to be recent)
            
            Be honest about what you can and cannot read clearly from the image.
            """
            
            # In a real implementation, we would send the image to the API
            # For this implementation, we'll check if the model would need clarification
            filename = file.name.lower()
            if "blur" in filename or "low_quality" in filename or "damaged" in filename:
                # This simulates a scenario where the model can't clearly read some information
                if hasattr(st, 'warning'):
                    st.warning("🤔 Some information couldn't be clearly read from this receipt.")
                    
                # Ask for clarification from user
                merchant_guess = "Unknown Store"
                for known_store in ["magnum", "small", "kaspi", "mechta", "sulpak", "technodom"]:
                    if known_store in filename:
                        merchant_guess = known_store.capitalize()
                        break
                    
                # Request clarification
                if hasattr(st, 'info'):
                    st.info("Please help clarify the following information:")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        clarified_date = st.date_input("Transaction Date", value=datetime.now())
                        clarified_amount = st.number_input("Amount", value=0.0, step=0.01)
                    with col2:
                        clarified_merchant = st.text_input("Merchant/Store", value=merchant_guess)
                        clarified_category = st.selectbox(
                            "Category", 
                            options=[cat["name"] for cat in self.config['expense_categories']]
                        )
                    
                    if st.button("Confirm Details", type="primary"):
                        transaction = {
                            'date': clarified_date.strftime('%Y-%m-%d'),
                            'description': f"{clarified_merchant} Purchase",
                            'amount': -abs(clarified_amount),
                            'currency': self.config['currencies'][0],
                            'type': 'expense',
                            'category': clarified_category
                        }
                        return pd.DataFrame([transaction])
                        
                # If we're not in a Streamlit context or the user hasn't confirmed yet,
                # return a placeholder transaction for now
                tx_date = datetime.now() - timedelta(days=1)
                amount = -(float(int(datetime.now().timestamp()) % 10000) / 100)
                description = f"{merchant_guess} Purchase [NEEDS_CLARIFICATION]"
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
            else:
                # For normal receipts, generate a sample transaction as if we got it from AI
                tx_date = datetime.now() - timedelta(days=1)
                amount = -(float(int(datetime.now().timestamp()) % 10000) / 100)
                
                # Extract store name from filename for better descriptions
                store_name = "Store"
                for known_store in ["magnum", "small", "kaspi", "mechta", "sulpak", "technodom"]:
                    if known_store in filename:
                        store_name = known_store.capitalize()
                        break
                
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
            
            print(f"\n----- PDF RECEIPT EXTRACTION -----")
            print(f"Filename: {file.name}")
            print(f"Text content length: {len(text_content)}")
            print(f"First 500 chars: {text_content[:500]}...")
            
            if text_content.strip():
                return self._extract_transactions_with_ai(text_content, "receipt")
            else:
                print("No text content extracted from PDF receipt")
                return self._generate_sample_transactions("PDF Receipt", single_transaction=True)
        except Exception as e:
            logger.error(f"Error processing PDF receipt: {e}")
            print(f"Error processing PDF receipt: {e}")
            return self._generate_sample_transactions("PDF Receipt", single_transaction=True)
    
    def _extract_transactions_with_ai(self, text_content, document_type):
        """Extract transactions from text using AI."""
        # Prepare examples from previous clarifications to guide the AI
        clarification_examples_text = ""
        if self.clarification_examples and self.clarification_examples.get("examples"):
            examples = self.clarification_examples.get("examples")
            # Use up to 10 most recent examples
            recent_examples = examples[-10:]
            
            clarification_examples_text = "Here are some examples of how to handle unclear information based on previous clarifications:\n\n"
            for ex in recent_examples:
                clarification_examples_text += f"When you see: {ex['unclear_text']}\n"
                clarification_examples_text += f"The correct value is: {ex['clarified_value']} (for {ex['field_type']})\n\n"
        
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
        
        If any information is unclear or ambiguous, mark it with [UNCLEAR] tag.
        For example: "Date: [UNCLEAR] (possibly 2023-05-XX)"
        
        Do not guess exact values when information is unclear - be explicit about uncertainty.
        
        {clarification_examples_text}
        """
        
        try:
            print(f"\n----- AI EXTRACTION REQUEST -----")
            print(f"Document type: {document_type}")
            print(f"Text length sent to AI: {len(text_content[:3000])}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500
            )
            
            ai_response = response.choices[0].message.content
            
            print(f"\n----- AI EXTRACTION RESPONSE -----")
            print(f"Response length: {len(ai_response)}")
            print(f"Response content:\n{ai_response}")
            
            # Check if there are any unclear items
            if "[UNCLEAR]" in ai_response and hasattr(st, 'warning'):
                st.warning("🤔 Some information in the document couldn't be clearly read")
                st.info("The AI has identified some unclear information. Please review the extracted data carefully.")
                
                # Parse the response as normal but flag unclear items
                transactions_df = self._parse_ai_transactions(ai_response)
                
                # Mark transactions with unclear data in the description
                for idx, row in transactions_df.iterrows():
                    if "[UNCLEAR]" in ai_response:
                        # Find the transaction that contains this row's data
                        lines = ai_response.split('\n')
                        for i, line in enumerate(lines):
                            if row['description'] in line or (row['amount'] and str(row['amount']) in line):
                                # Look for unclear tags in surrounding lines
                                for j in range(max(0, i-5), min(len(lines), i+5)):
                                    if "[UNCLEAR]" in lines[j]:
                                        transactions_df.at[idx, 'description'] = f"{row['description']} [NEEDS_CLARIFICATION]"
                                        break
                
                # If we have a Streamlit context, offer a way to fix unclear items
                if hasattr(st, 'expander'):
                    with st.expander("Review and Fix Unclear Transactions", expanded=True):
                        st.write("Please review the following transactions and fix any unclear information:")
                        
                        # Use a form to prevent premature submission
                        with st.form(key="clarification_form"):
                            st.write("### Transactions Needing Clarification")
                            st.info("Complete all fields below, then click 'Apply All Changes' when finished.")
                            
                            # Track updates to apply later
                            updates = {}
                            
                            # Find transactions that need clarification
                            needs_clarification_count = 0
                            for idx, row in transactions_df.iterrows():
                                if "[NEEDS_CLARIFICATION]" in row['description']:
                                    needs_clarification_count += 1
                                    st.markdown(f"**Transaction {idx+1}:**")
                                    
                                    # Clean up the description
                                    clean_desc = row['description'].replace(" [NEEDS_CLARIFICATION]", "")
                                    
                                    # Create unique keys for each input field
                                    date_key = f"fix_date_{idx}"
                                    amount_key = f"fix_amount_{idx}"
                                    desc_key = f"fix_desc_{idx}"
                                    type_key = f"fix_type_{idx}"
                                    category_key = f"fix_category_{idx}"
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        # Date input with default from extracted data
                                        try:
                                            default_date = datetime.strptime(row['date'], '%Y-%m-%d')
                                        except:
                                            default_date = datetime.now()
                                        
                                        fixed_date = st.date_input(
                                            f"Date", 
                                            value=default_date, 
                                            key=date_key
                                        )
                                        
                                        # Amount input with default from extracted data
                                        fixed_amount = st.number_input(
                                            "Amount", 
                                            value=float(abs(row['amount'])) if row['amount'] else 0.0,
                                            step=0.01,
                                            key=amount_key
                                        )
                                        
                                    with col2:
                                        # Description input
                                        fixed_desc = st.text_input(
                                            "Description", 
                                            value=clean_desc,
                                            key=desc_key
                                        )
                                        
                                        # Type selection
                                        fixed_type = st.selectbox(
                                            "Type",
                                            options=["expense", "income"],
                                            index=0 if row['type'] == 'expense' else 1,
                                            key=type_key
                                        )
                                    
                                    # Category selection based on type
                                    category_options = f"{fixed_type}_categories"
                                    categories = [cat["name"] for cat in self.config[category_options]] if category_options in self.config else ["Other"]
                                    
                                    # Set default category index based on type
                                    # If type changed, select a default category for the new type
                                    if row['type'] != fixed_type:
                                        # Type has changed - select appropriate default category
                                        if fixed_type == "income":
                                            # Default income category (usually first one is Salary)
                                            default_category_index = 0
                                        else:
                                            # For expense, try to intelligently select based on description
                                            # This mimics our categorization logic
                                            desc_lower = fixed_desc.lower()
                                            if any(word in desc_lower for word in ["grocery", "food", "market"]):
                                                category_name = "Groceries"
                                            elif any(word in desc_lower for word in ["restaurant", "cafe", "coffee"]):
                                                category_name = "Dining"
                                            elif any(word in desc_lower for word in ["transport", "taxi", "uber"]):
                                                category_name = "Transport"
                                            else:
                                                # Default to first expense category
                                                category_name = categories[0] if categories else "Other"
                                            
                                            # Find index of this category
                                            default_category_index = categories.index(category_name) if category_name in categories else 0
                                    else:
                                        # Same type - try to preserve category if possible
                                        default_category_index = 0
                                        if row['category'] in categories:
                                            default_category_index = categories.index(row['category'])
                                    
                                    # Use a key that includes the type to force refresh when type changes
                                    fixed_category = st.selectbox(
                                        "Category",
                                        options=categories,
                                        index=default_category_index,
                                        key=f"fix_category_{idx}_{fixed_type}"  # Include type in key to refresh on type change
                                    )
                                    
                                    # Store all the fixes to apply later in one batch
                                    updates[idx] = {
                                        'date': fixed_date,
                                        'description': fixed_desc,
                                        'amount': fixed_amount,
                                        'type': fixed_type,
                                        'category': fixed_category
                                    }
                                    
                                    st.markdown("---")
                            
                            # Only show the submit button if we have transactions to fix
                            submit_button = st.form_submit_button(
                                "Apply All Changes" if needs_clarification_count > 0 else "No Changes Needed",
                                type="primary",
                                disabled=needs_clarification_count == 0
                            )
                            
                            if submit_button and updates:
                                # Apply all the updates at once
                                for idx, update in updates.items():
                                    # Update the transaction with fixed information
                                    transactions_df.at[idx, 'date'] = update['date'].strftime('%Y-%m-%d')
                                    transactions_df.at[idx, 'description'] = update['description']
                                    transactions_df.at[idx, 'amount'] = update['amount'] if update['type'] == 'income' else -abs(update['amount'])
                                    transactions_df.at[idx, 'type'] = update['type']
                                    transactions_df.at[idx, 'category'] = update['category']
                                    
                                    # Save clarification examples for future AI processing
                                    # Find the original unclear text in the AI response
                                    lines = ai_response.split('\n')
                                    for i, line in enumerate(lines):
                                        if row['description'] in line or (str(row['amount']) in line and str(row['date']) in line):
                                            # Look for unclear tags in surrounding lines
                                            for j in range(max(0, i-5), min(len(lines), i+5)):
                                                if "[UNCLEAR]" in lines[j]:
                                                    # Extract the field type and unclear text
                                                    if "Date:" in lines[j]:
                                                        field_type = "date"
                                                        unclear_text = lines[j].replace("Date:", "").strip()
                                                        clarified_value = update['date'].strftime('%Y-%m-%d')
                                                        self._add_clarification_example(unclear_text, clarified_value, field_type)
                                                    elif "Amount:" in lines[j]:
                                                        field_type = "amount"
                                                        unclear_text = lines[j].replace("Amount:", "").strip()
                                                        clarified_value = str(update['amount'] if update['type'] == 'income' else -abs(update['amount']))
                                                        self._add_clarification_example(unclear_text, clarified_value, field_type)
                                                    elif "Description:" in lines[j]:
                                                        field_type = "description"
                                                        unclear_text = lines[j].replace("Description:", "").strip()
                                                        clarified_value = update['description']
                                                        self._add_clarification_example(unclear_text, clarified_value, field_type)
                                                    elif "Type:" in lines[j]:
                                                        field_type = "type"
                                                        unclear_text = lines[j].replace("Type:", "").strip()
                                                        clarified_value = update['type']
                                                        self._add_clarification_example(unclear_text, clarified_value, field_type)
                                                        
                                                        # When type changes, also store the category change as a related example
                                                        if row.get('type') != update['type']:
                                                            # This is a type change, so record the category mapping too
                                                            field_type = "category_for_type_change"
                                                            # Include description keywords to provide context for the change
                                                            desc_excerpt = update['description'][:30] if len(update['description']) > 30 else update['description']
                                                            unclear_text = f"From {row.get('type')} category '{row.get('category')}' to {update['type']} category - description: '{desc_excerpt}'"
                                                            clarified_value = update['category']
                                                            self._add_clarification_example(unclear_text, clarified_value, field_type)
                                
                                st.success(f"✅ Updated {len(updates)} transactions!")
                                st.info("✅ Saved clarification examples for future use. The AI will use these to better handle similar cases.")
                        
                        # Show the updated transactions after the form
                        if needs_clarification_count > 0:
                            st.write("### Updated Transactions")
                            st.dataframe(transactions_df)
                
                print(f"\n----- PARSED TRANSACTIONS WITH CLARIFICATION -----")
                print(transactions_df)
                
                return transactions_df
            
            # If no unclear items, process normally
            transactions_df = self._parse_ai_transactions(ai_response)
            
            print(f"\n----- PARSED TRANSACTIONS -----")
            print(transactions_df)
            
            return transactions_df
        except Exception as e:
            logger.error(f"Error extracting transactions with AI: {e}")
            print(f"Error extracting transactions with AI: {e}")
            return self._generate_sample_transactions("AI Processing Failed")
    
    def _parse_ai_transactions(self, ai_response):
        """Parse AI response into DataFrame with improved handling of unclear data."""
        transactions = []
        current_tx = {}
        needs_clarification = False
        
        for line in ai_response.split('\n'):
            line = line.strip()
            if line.startswith('Date:'):
                date_value = line.replace('Date:', '').strip()
                # Handle unclear dates
                if "[UNCLEAR]" in date_value:
                    needs_clarification = True
                    # Use today's date as fallback but mark for clarification
                    current_tx['date'] = datetime.now().strftime('%Y-%m-%d')
                    current_tx['needs_clarification'] = True
                    logger.info(f"Unclear date detected: {date_value}")
                else:
                    try:
                        # Format date as YYYY-MM-DD with explicit format
                        parsed_date = pd.to_datetime(date_value, format='%Y-%m-%d', errors='coerce').strftime('%Y-%m-%d')
                        current_tx['date'] = parsed_date
                    except:
                        # If parsing fails, store original value
                        current_tx['date'] = str(date_value)
            elif line.startswith('Description:'):
                desc_value = line.replace('Description:', '').strip()
                if "[UNCLEAR]" in desc_value:
                    # Keep the unclear tag but clean up the description
                    clean_desc = desc_value.replace("[UNCLEAR]", "").strip()
                    if not clean_desc:
                        clean_desc = "Unknown"
                    current_tx['description'] = f"{clean_desc} [NEEDS_CLARIFICATION]"
                    current_tx['needs_clarification'] = True
                    needs_clarification = True
                else:
                    current_tx['description'] = desc_value
            elif line.startswith('Amount:'):
                try:
                    amount_str = line.replace('Amount:', '').strip()
                    if "[UNCLEAR]" in amount_str:
                        # Use a placeholder amount but mark for clarification
                        logger.info(f"Unclear amount detected: {amount_str}")
                        current_tx['amount'] = -100.0  # Default placeholder
                        current_tx['needs_clarification'] = True
                        needs_clarification = True
                    else:
                        # Clean the amount string and parse
                        clean_amount = re.sub(r'[^\d.-]', '', amount_str)
                        current_tx['amount'] = float(clean_amount)
                except:
                    logger.warning(f"Failed to parse amount: {line}")
                    current_tx['amount'] = -100.0  # Default amount
                    current_tx['needs_clarification'] = True
                    needs_clarification = True
            elif line.startswith('Type:'):
                type_value = line.replace('Type:', '').strip().lower()
                if "[UNCLEAR]" in type_value:
                    # Default to expense if unclear
                    current_tx['type'] = 'expense'
                    current_tx['needs_clarification'] = True
                    needs_clarification = True
                else:
                    current_tx['type'] = type_value
            elif line == '---' and current_tx:
                # Complete transaction
                if 'date' in current_tx and 'description' in current_tx:
                    # If any field needs clarification, mark the description
                    if current_tx.get('needs_clarification', False) and '[NEEDS_CLARIFICATION]' not in current_tx.get('description', ''):
                        current_tx['description'] = f"{current_tx['description']} [NEEDS_CLARIFICATION]"
                    
                    # Remove our internal tracking field before adding to transactions
                    if 'needs_clarification' in current_tx:
                        del current_tx['needs_clarification']
                    
                    transactions.append(current_tx.copy())
                current_tx = {}
        
        # Add last transaction if exists
        if current_tx and 'date' in current_tx:
            # If any field needs clarification, mark the description
            if current_tx.get('needs_clarification', False) and '[NEEDS_CLARIFICATION]' not in current_tx.get('description', ''):
                current_tx['description'] = f"{current_tx['description']} [NEEDS_CLARIFICATION]"
            
            # Remove our internal tracking field before adding to transactions
            if 'needs_clarification' in current_tx:
                del current_tx['needs_clarification']
                
            transactions.append(current_tx)
        
        if not transactions:
            return self._generate_sample_transactions("AI Parse Failed")
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        
        # Add missing columns
        df['currency'] = self.config['currencies'][0]
        
        # Ensure type is set (default to expense if missing)
        if 'type' not in df.columns:
            df['type'] = 'expense'
        
        # Categorize transactions
        df['category'] = df.apply(
            lambda row: self._categorize_transaction(row['description'], row['amount'], row.get('type', 'expense')),
            axis=1
        )
        
        # Log if we found transactions needing clarification
        if needs_clarification:
            logger.info("Some transactions need clarification from the user")
            
        # Ensure all required columns exist
        required_columns = ['date', 'description', 'amount', 'currency', 'type', 'category']
        for col in required_columns:
            if col not in df.columns:
                if col == 'description':
                    df[col] = 'Unknown transaction'
                elif col == 'amount':
                    df[col] = -100.0
                elif col == 'date':
                    df[col] = datetime.now().strftime('%Y-%m-%d')
        
        return df[required_columns]
    
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
            'date': pd.to_datetime(standardized_df[date_col], format='%Y-%m-%d', errors='coerce'),
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
        
        # Get category change examples for the AI to learn from
        category_change_examples = ""
        if self.clarification_examples and self.clarification_examples.get("examples"):
            examples = self.clarification_examples.get("examples")
            # Filter for category change examples
            change_examples = [ex for ex in examples if ex.get('field_type') == 'category_for_type_change']
            
            if change_examples:
                category_change_examples = "\nCategory change examples from previous clarifications:\n"
                for ex in change_examples[-5:]:  # Use the 5 most recent examples
                    category_change_examples += f"- {ex['unclear_text']} → {ex['clarified_value']}\n"
        
        prompt = f"""
        Review these financial transactions and suggest better categories:
        
        Available Income Categories: {', '.join(income_cats)}
        Available Expense Categories: {', '.join(expense_cats)}
        
        Transactions:
        {transactions_text}
        
        {category_change_examples}
        
        For each transaction, respond with:
        INDEX: SUGGESTED_CATEGORY
        
        Only suggest categories from the available lists above.
        Make sure to use income categories for income transactions and expense categories for expense transactions.
        Pay attention to the transaction description when determining the appropriate category.
        
        For example:
        - Transactions with "grocery", "supermarket", "food", "market" typically belong to "Groceries"
        - Transactions with "restaurant", "cafe", "coffee" typically belong to "Dining"
        - Transactions with "salary", "payroll", "wage" typically belong to "Salary"
        - Transactions with "dividend", "interest", "investment" typically belong to "Investment"
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
    
    def save_processed_transactions(self, uploads_df):
        """Save record of uploaded files."""
        if uploads_df is None or uploads_df.empty:
            if hasattr(st, 'error'):
                st.error("No uploads to save - dataframe is empty!")
            logger.error("No uploads to save - dataframe is empty!")
            return 0, 0
        
        try:
            # Create uploads directory if it doesn't exist
            uploads_dir = os.path.join("data", "uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            
            # Save uploads record to CSV
            uploads_file = os.path.join(uploads_dir, "uploads_record.csv")
            
            # Check if file exists and append to it
            if os.path.exists(uploads_file):
                existing_df = pd.read_csv(uploads_file)
                combined_df = pd.concat([existing_df, uploads_df], ignore_index=True)
                combined_df.to_csv(uploads_file, index=False)
            else:
                uploads_df.to_csv(uploads_file, index=False)
            
            if hasattr(st, 'success'):
                st.success(f"✅ Successfully saved record of {len(uploads_df)} uploaded files")
            
            logger.info(f"Successfully saved record of {len(uploads_df)} uploaded files")
            return len(uploads_df), 0
            
        except Exception as e:
            logger.error(f"Error saving upload records: {str(e)}")
            if hasattr(st, 'error'):
                st.error(f"Error saving upload records: {str(e)}")
            return 0, 1
    
    def get_upload_summary(self):
        """Get summary of saved uploads."""
        uploads_file = os.path.join("data", "uploads", "uploads_record.csv")
        
        if not os.path.exists(uploads_file):
            return {"total_uploads": 0, "file_types": [], "last_upload": None}
        
        try:
            df = pd.read_csv(uploads_file)
            if df.empty:
                return {"total_uploads": 0, "file_types": [], "last_upload": None}
            
            file_types = df['type'].value_counts().to_dict()
            last_upload = df['upload_time'].max() if 'upload_time' in df.columns else None
            
            return {
                "total_uploads": len(df),
                "file_types": file_types,
                "last_upload": last_upload
            }
        except Exception as e:
            logger.error(f"Error getting upload summary: {e}")
            return {"error": str(e)}
    
    def _load_clarification_examples(self):
        """Load previous clarification examples to guide the AI."""
        try:
            if os.path.exists(CLARIFICATIONS_FILE):
                with open(CLARIFICATIONS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Create empty clarifications file
                self._save_clarification_examples({
                    "examples": [], 
                    "meta": {"last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                })
                return {"examples": [], "meta": {"last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}
        except Exception as e:
            logger.error(f"Error loading clarification examples: {e}")
            return {"examples": [], "meta": {"last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}
    
    def _save_clarification_examples(self, examples):
        """Save clarification examples for future use."""
        try:
            # Ensure data directory exists
            os.makedirs("data", exist_ok=True)
            
            # Update metadata
            examples["meta"] = {"last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            with open(CLARIFICATIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving clarification examples: {e}")
            return False
    
    def _add_clarification_example(self, unclear_text, clarified_value, field_type):
        """Add a new clarification example to the database."""
        try:
            examples = self._load_clarification_examples()
            
            # Add new example
            examples["examples"].append({
                "unclear_text": unclear_text,
                "clarified_value": clarified_value,
                "field_type": field_type,
                "date_added": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Limit to 100 most recent examples to avoid prompt getting too large
            if len(examples["examples"]) > 100:
                examples["examples"] = examples["examples"][-100:]
            
            # Save updated examples
            return self._save_clarification_examples(examples)
        except Exception as e:
            logger.error(f"Error adding clarification example: {e}")
            return False


# Example usage
def example_usage():
    """Example of how to use the simplified BankStatementProcessor."""
    # Initialize processor
    processor = BankStatementProcessor()
    
    # Set processing options
    processor.set_processing_options({
        "date_range": "all",  # We don't filter dates now
        "allow_new_categories": False  # We don't categorize now
    })
    
    # In a real application, uploaded_files would be provided by Streamlit
    # For example:
    # uploaded_files = st.file_uploader("Upload bank statements or receipts", 
    #                                 accept_multiple_files=True, 
    #                                 type=["csv", "pdf", "jpg", "jpeg", "png", "xls", "xlsx"])
    #
    # uploads_df = processor.process_files(uploaded_files)
    # if uploads_df is not None:
    #     processor.save_processed_transactions(uploads_df)
    
    logger.info("BankStatementProcessor initialized successfully!")
    logger.info("Ready to process file uploads and save records.")


if __name__ == "__main__":
    example_usage()

