"""
Transaction Processor
Processes bank statements and receipts in various formats (CSV, PDF, XLSX, PNG, JPG)
using OpenAI to categorize them based on config.
"""
import os
import re
import base64
import io
import tempfile
from datetime import datetime, timedelta
import json
import logging
from typing import List, Dict, Any, Union, Optional

import pandas as pd
import numpy as np
import pdfplumber
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Required for improved document processing
try:
    # PDF processing
    import fitz  # PyMuPDF
    import tabula
    
    # Image processing
    import cv2
    import pytesseract
    from PIL import Image
    
    # Set tesseract path if needed (uncomment and set path if not in PATH)
    # pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
    
    ENHANCED_PROCESSING = True
    logger.info("Enhanced document processing enabled with PyMuPDF, tabula-py, and OCR")
except ImportError as e:
    logger.warning(f"Some document processing libraries are missing: {e}")
    logger.warning("Basic document processing will be used")
    ENHANCED_PROCESSING = False

class TransactionProcessor:
    """Processes financial documents and extracts transactions with categorization."""
    
    def __init__(self, api_key=None, config_path=None):
        """Initialize with OpenAI API key and config."""
        # Get API key from arguments, environment, or streamlit secrets
        self.api_key = api_key
        
        if not self.api_key:
            # Try to get from environment
            self.api_key = os.getenv('OPENAI_API_KEY')
            
            # If still not found, try to get from streamlit secrets
            if not self.api_key and os.path.exists('.streamlit/secrets.toml'):
                import toml
                try:
                    secrets = toml.load('.streamlit/secrets.toml')
                    self.api_key = secrets.get('OPENAI_API_KEY')
                except Exception as e:
                    logger.error(f"Error loading secrets: {e}")
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in environment or pass as argument.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4o-mini"  # You can change this to a different model
        
        # Load configuration
        self.config_path = config_path or "data/config.json"
        self.config = self._load_config()
        
        # Set up processing options
        self.processing_options = {
            "date_range": "all",
            "allow_new_categories": True
        }
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Return default config
                return {
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
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def process_file(self, file_path: str) -> dict:
        """Process a financial document file and extract transactions.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            dict: Dictionary with success status and extracted transactions
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"success": False, "error": "File not found", "transactions": []}
        
        logger.info(f"Processing file: {file_path}")
        
        try:
            # Determine file type by extension
            file_extension = os.path.splitext(file_path)[1].lower()
            
            # Extract text content based on file type
            text_content = None
            extracted_df = None
            
            # Process file based on type
            if file_extension in ['.csv']:
                # Process CSV directly with Python libraries
                extracted_df = self._process_csv(file_path)
                if not extracted_df.empty:
                    logger.info(f"Successfully extracted data from CSV: {len(extracted_df)} rows")
            
            elif file_extension in ['.xlsx', '.xls']:
                # Process Excel directly with Python libraries
                extracted_df = self._process_excel(file_path)
                if not extracted_df.empty:
                    logger.info(f"Successfully extracted data from Excel: {len(extracted_df)} rows")
            
            elif file_extension in ['.pdf']:
                # Extract text from PDF using Python libraries
                text_content = self._extract_text_from_pdf(file_path)
                logger.info(f"Extracted {len(text_content) if text_content else 0} characters from PDF")
            
            elif file_extension in ['.jpg', '.jpeg', '.png']:
                # Extract text from image using OCR if available, otherwise describe the image
                if ENHANCED_PROCESSING:
                    text_content = self._extract_text_from_image(file_path)
                    logger.info(f"Extracted {len(text_content) if text_content else 0} characters from image using OCR")
                else:
                    # For images without OCR capability, generate a description
                    text_content = self._describe_image(file_path)
                    logger.info(f"Generated {len(text_content) if text_content else 0} characters description for image")
            
            else:
                logger.error(f"Unsupported file type: {file_extension}")
                return {"success": False, "error": f"Unsupported file type: {file_extension}", "transactions": []}
            
            # If we have a DataFrame already, use it directly
            if extracted_df is not None and not extracted_df.empty:
                df = extracted_df
            # If we have text content, extract transactions using OpenAI
            elif text_content:
                if not text_content.strip():
                    logger.warning(f"No text content extracted from file: {file_path}")
                    return {"success": True, "transactions": []}
                
                df = self._extract_transactions_with_ai(text_content)
            else:
                logger.warning(f"Could not extract content from file: {file_path}")
                return {"success": False, "error": "Could not extract content from file", "transactions": []}
            
            # Return empty list if no transactions found
            if df.empty:
                logger.warning(f"No transactions found in file: {file_path}")
                return {"success": True, "transactions": []}
            
            # Convert DataFrame to list of dictionaries
            transactions = df.to_dict('records')
            
            logger.info(f"Extracted {len(transactions)} transactions from file")
            return {"success": True, "transactions": transactions}
        
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e), "transactions": []}
    
    def _process_csv(self, file_path: str) -> pd.DataFrame:
        """Process CSV file with improved format detection."""
        try:
            # Try to auto-detect delimiter
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                sample = f.read(5000)  # Read a sample to detect delimiter
                
            if ',' in sample:
                delimiter = ','
            elif ';' in sample:
                delimiter = ';'
            elif '\t' in sample:
                delimiter = '\t'
            else:
                delimiter = None  # Let pandas autodetect
                
            logger.info(f"Reading CSV with delimiter: {delimiter}")
            
            # Try to read CSV with detected delimiter
            df = pd.read_csv(file_path, delimiter=delimiter, encoding='utf-8', errors='replace')
            
            # If the CSV looks like a standard bank statement, try direct mapping
            column_names = [col.lower() for col in df.columns]
            
            # Check if this looks like a bank statement with common columns
            date_cols = [col for col in column_names if any(date_term in col for date_term in ['date', 'time', 'дата'])]
            amount_cols = [col for col in column_names if any(amount_term in col for amount_term in ['amount', 'sum', 'сумма'])]
            desc_cols = [col for col in column_names if any(desc_term in col for desc_term in ['desc', 'narration', 'detail', 'merchant', 'назначение'])]
            
            if date_cols and (amount_cols or desc_cols):
                logger.info("CSV appears to be a bank statement, trying direct mapping")
                
                # Map columns to our format
                transactions = []
                
                for _, row in df.iterrows():
                    tx = {}
                    
                    # Try to get date
                    if date_cols:
                        date_val = row[df.columns[column_names.index(date_cols[0])]]
                        try:
                            # Try different date formats
                            date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d.%m.%Y', '%Y/%m/%d']
                            for fmt in date_formats:
                                try:
                                    tx['date'] = pd.to_datetime(date_val, format=fmt).strftime('%Y-%m-%d')
                                    break
                                except:
                                    continue
                        except:
                            tx['date'] = f"[UNCLEAR] {date_val}"
                    
                    # Try to get description
                    if desc_cols:
                        tx['description'] = str(row[df.columns[column_names.index(desc_cols[0])]])
                    
                    # Try to get amount
                    if amount_cols:
                        amount_val = row[df.columns[column_names.index(amount_cols[0])]]
                        try:
                            tx['amount'] = float(str(amount_val).replace(',', '.').replace(' ', ''))
                        except:
                            tx['amount'] = f"[UNCLEAR] {amount_val}"
                    
                    # Determine transaction type
                    if 'amount' in tx and not isinstance(tx['amount'], str):
                        tx['type'] = 'income' if tx['amount'] > 0 else 'expense'
                    else:
                        tx['type'] = 'expense'  # Default
                    
                    # Add transaction if it has required fields
                    if ('date' in tx and 'description' in tx) or ('date' in tx and 'amount' in tx):
                        transactions.append(tx)
                
                if transactions:
                    logger.info(f"Extracted {len(transactions)} transactions directly from CSV")
                    return pd.DataFrame(transactions)
            
            # If direct mapping didn't work or no transactions found, use AI
            logger.info("Converting CSV to text for AI processing")
            csv_text = df.to_string(index=False)
            return self._extract_transactions_with_ai(csv_text)
        
        except Exception as e:
            logger.error(f"Error processing CSV file: {e}")
            # Try to read the file as text and process with AI as fallback
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    text_content = f.read()
                return self._extract_transactions_with_ai(text_content)
            except Exception as inner_e:
                logger.error(f"Error in CSV fallback processing: {inner_e}")
                return pd.DataFrame()
    
    def _process_excel(self, file_path: str) -> pd.DataFrame:
        """Process Excel file with improved sheet handling."""
        try:
            # Load the Excel file
            logger.info(f"Reading Excel file: {file_path}")
            
            # Read all sheets to find transaction data
            excel = pd.ExcelFile(file_path)
            sheet_names = excel.sheet_names
            logger.info(f"Found {len(sheet_names)} sheets: {sheet_names}")
            
            all_transactions = []
            
            for sheet_name in sheet_names:
                logger.info(f"Processing sheet: {sheet_name}")
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Skip empty sheets
                if df.empty:
                    logger.info(f"Sheet {sheet_name} is empty, skipping")
                    continue
                    
                # Check if this looks like a transaction sheet
                column_names = [str(col).lower() for col in df.columns]
                
                date_cols = [col for col in column_names if any(date_term in col for date_term in ['date', 'time', 'дата'])]
                amount_cols = [col for col in column_names if any(amount_term in col for amount_term in ['amount', 'sum', 'сумма'])]
                desc_cols = [col for col in column_names if any(desc_term in col for desc_term in ['desc', 'narration', 'detail', 'merchant', 'назначение'])]
                
                if date_cols and (amount_cols or desc_cols):
                    logger.info(f"Sheet {sheet_name} appears to contain transaction data")
                    
                    # Map columns to our format
                    sheet_transactions = []
                    
                    for _, row in df.iterrows():
                        tx = {}
                        
                        # Try to get date
                        if date_cols:
                            date_val = row[df.columns[column_names.index(date_cols[0])]]
                            try:
                                # Handle different date formats
                                if isinstance(date_val, (datetime.datetime, datetime.date)):
                                    tx['date'] = date_val.strftime('%Y-%m-%d')
                                else:
                                    # Try parsing as string with different formats
                                    date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d.%m.%Y', '%Y/%m/%d']
                                    for fmt in date_formats:
                                        try:
                                            tx['date'] = pd.to_datetime(date_val, format=fmt).strftime('%Y-%m-%d')
                                            break
                                        except:
                                            continue
                                            
                                    if 'date' not in tx:
                                        tx['date'] = f"[UNCLEAR] {date_val}"
                            except:
                                tx['date'] = f"[UNCLEAR] {date_val}"
                        
                        # Try to get description
                        if desc_cols:
                            tx['description'] = str(row[df.columns[column_names.index(desc_cols[0])]])
                        
                        # Try to get amount
                        if amount_cols:
                            amount_val = row[df.columns[column_names.index(amount_cols[0])]]
                            try:
                                if pd.isna(amount_val):
                                    continue  # Skip rows with no amount
                                tx['amount'] = float(str(amount_val).replace(',', '.').replace(' ', ''))
                            except:
                                tx['amount'] = f"[UNCLEAR] {amount_val}"
                        
                        # Determine transaction type
                        if 'amount' in tx and not isinstance(tx['amount'], str):
                            tx['type'] = 'income' if tx['amount'] > 0 else 'expense'
                        else:
                            tx['type'] = 'expense'  # Default
                        
                        # Add transaction if it has required fields
                        if ('date' in tx and 'description' in tx) or ('date' in tx and 'amount' in tx):
                            sheet_transactions.append(tx)
                    
                    if sheet_transactions:
                        all_transactions.extend(sheet_transactions)
                
                # If no transactions found with direct mapping, try AI
                if not all_transactions:
                    logger.info(f"No structured transactions found in sheet {sheet_name}, trying AI extraction")
                    excel_text = df.to_string(index=False)
                    ai_transactions = self._extract_transactions_with_ai(excel_text)
                    if not ai_transactions.empty:
                        all_transactions.extend(ai_transactions.to_dict('records'))
            
            if all_transactions:
                logger.info(f"Extracted {len(all_transactions)} transactions from Excel")
                return pd.DataFrame(all_transactions)
            
            # If no transactions found in any sheet, try sending the whole Excel as text to AI
            all_text = ""
            for sheet_name in sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                all_text += f"\nSheet: {sheet_name}\n{df.to_string(index=False)}\n"
            
            return self._extract_transactions_with_ai(all_text)
        
        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            return pd.DataFrame()
    
    def _process_pdf(self, file_path: str) -> pd.DataFrame:
        """Process PDF file with improved text extraction."""
        try:
            logger.info(f"Reading PDF file: {file_path}")
            doc = fitz.open(file_path)
            
            # Check for text extraction first
            text_content = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text_content += page.get_text()
            
            logger.info(f"Extracted {len(text_content)} characters of text from PDF")
            
            # If PDF has very little text, it might be scanned - try OCR
            if len(text_content.strip()) < 100:
                logger.info("PDF appears to be scanned or has little text, attempting OCR")
                return self._process_image(file_path)
            
            # Try to find tables in the PDF
            tables_found = False
            
            try:
                logger.info("Attempting to extract tables from PDF")
                tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
                
                if tables and len(tables) > 0:
                    logger.info(f"Found {len(tables)} tables in PDF")
                    tables_found = True
                    
                    all_transactions = []
                    
                    for i, table in enumerate(tables):
                        if table.empty:
                            continue
                            
                        logger.info(f"Processing table {i+1} with columns: {table.columns}")
                        
                        # Check if this looks like a transaction table
                        column_names = [str(col).lower() for col in table.columns]
                        
                        date_cols = [col for col in column_names if any(date_term in col for date_term in ['date', 'time', 'дата'])]
                        amount_cols = [col for col in column_names if any(amount_term in col for amount_term in ['amount', 'sum', 'сумма'])]
                        desc_cols = [col for col in column_names if any(desc_term in col for date_term in ['desc', 'narration', 'detail', 'merchant', 'назначение'])]
                        
                        if date_cols or amount_cols or desc_cols:
                            # This looks like a transaction table, extract transactions
                            table_text = table.to_string(index=False)
                            table_transactions = self._extract_transactions_with_ai(table_text)
                            
                            if not table_transactions.empty:
                                all_transactions.extend(table_transactions.to_dict('records'))
                    
                    if all_transactions:
                        logger.info(f"Extracted {len(all_transactions)} transactions from PDF tables")
                        return pd.DataFrame(all_transactions)
            except Exception as table_error:
                logger.warning(f"Error extracting tables from PDF: {table_error}")
            
            # If no tables found or no transactions extracted from tables, use text content
            return self._extract_transactions_with_ai(text_content)
        
        except Exception as e:
            logger.error(f"Error processing PDF file: {e}")
            return pd.DataFrame()
    
    def _process_image(self, file_path: str) -> pd.DataFrame:
        """Process image file with OCR to extract text."""
        try:
            logger.info(f"Processing image file with OCR: {file_path}")
            
            # For PDFs, convert to images first
            if file_path.lower().endswith('.pdf'):
                import fitz
                import tempfile
                
                # Create a temporary directory for image files
                with tempfile.TemporaryDirectory() as temp_dir:
                    doc = fitz.open(file_path)
                    image_files = []
                    
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                        image_file = f"{temp_dir}/page_{page_num+1}.png"
                        pix.save(image_file)
                        image_files.append(image_file)
                    
                    # Process each image and combine results
                    all_text = ""
                    for img_file in image_files:
                        all_text += self._extract_text_from_image(img_file) + "\n\n"
                    
                    return self._extract_transactions_with_ai(all_text)
            else:
                # Process regular image file
                text = self._extract_text_from_image(file_path)
                return self._extract_transactions_with_ai(text)
        
        except Exception as e:
            logger.error(f"Error processing image file: {e}")
            return pd.DataFrame()
    
    def _extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using OCR with preprocessing for better results."""
        try:
            # Read the image
            img = cv2.imread(image_path)
            
            # Apply preprocessing to improve OCR results
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to get black and white image
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Reduce noise
            denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
            
            # OCR with multiple languages for better results
            text = pytesseract.image_to_string(denoised, lang='eng+rus', config='--psm 6')
            
            logger.info(f"Extracted {len(text)} characters from image using OCR")
            return text
        
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return ""
    
    def _extract_transactions_with_ai(self, text_content: str) -> pd.DataFrame:
        """Extract transactions from text using OpenAI with improved prompting."""
        # Prepare expense and income categories from config
        expense_categories = [cat["name"] for cat in self.config.get("expense_categories", [])]
        income_categories = [cat["name"] for cat in self.config.get("income_categories", [])]
        all_categories = expense_categories + income_categories
        
        # Create prompt with available categories and more detailed instructions
        prompt = f"""
        You are a financial document analyzer. Your task is to extract transaction information from text and format it in a structured way.
        
        Extract ALL financial transactions from this text. For each transaction, include:
        1. Date (format as YYYY-MM-DD)
        2. Description (merchant name, purpose of transaction)
        3. Amount (use negative numbers for expenses, positive for income)
        4. Type (categorize as either 'income' or 'expense')
        5. Category (choose from the categories below)
        
        Available Categories:
        - Income Categories: {', '.join(income_categories)}
        - Expense Categories: {', '.join(expense_categories)}
        
        FORMAT EACH TRANSACTION LIKE THIS EXACTLY:
        TRANSACTION:
        Date: YYYY-MM-DD
        Description: [merchant/description]
        Amount: [amount with + or - sign]
        Type: [income/expense]
        Category: [appropriate category]
        ---
        
        If any information is unclear or ambiguous, mark it with [UNCLEAR]. Example: "Date: [UNCLEAR] (possibly 2023-05-XX)"
        
        DO NOT omit any transactions, even if information is partial.
        DO NOT add explanatory text outside the transaction blocks.
        DO NOT skip transactions even if they seem unimportant.
        MAINTAIN the exact format specified above.
        
        Text content:
        {text_content[:4000]}
        """
        
        try:
            logger.info("Sending text to OpenAI for transaction extraction")
            
            # Use a structured format request to get better results
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial document analyzer that extracts transaction data in a structured format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            ai_response = response.choices[0].message.content
            logger.info(f"Received response from OpenAI: {len(ai_response)} characters")
            
            # Log a sample of the response for debugging
            logger.debug(f"Sample response: {ai_response[:200]}...")
            
            # Parse AI response into DataFrame
            return self._parse_ai_transactions(ai_response)
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return pd.DataFrame()
    
    def _parse_ai_transactions(self, ai_response: str) -> pd.DataFrame:
        """Parse AI response into DataFrame of transactions with improved robustness."""
        transactions = []
        current_tx = {}
        in_transaction = False
        
        # Debug the AI response
        logger.info(f"Parsing AI response with {len(ai_response)} characters")
        logger.debug(f"AI response: {ai_response[:500]}...")  # Log first 500 chars
        
        # First, check for common markers of transaction blocks
        transaction_blocks = []
        
        # Try to extract transaction blocks using different patterns
        if "TRANSACTION:" in ai_response:
            # Use the explicit TRANSACTION marker
            blocks = ai_response.split("TRANSACTION:")
            # Skip the first block if it's just introductory text
            transaction_blocks = ["TRANSACTION:" + block for block in blocks[1:]]
        elif "Transaction " in ai_response:
            # Alternative format like "Transaction 1:", "Transaction 2:"
            import re
            blocks = re.split(r'Transaction \d+:', ai_response)
            transaction_blocks = ["Transaction:" + block for block in blocks[1:]]
        else:
            # If no explicit transaction markers, look for date patterns as transaction starters
            import re
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}\.\d{2}\.\d{4}'  # DD.MM.YYYY
            ]
            
            pattern = '|'.join(f"({p})" for p in date_patterns)
            parts = re.split(f"(^|\\n)({pattern})", ai_response)
            
            # Reconstruct potential transaction blocks
            i = 0
            while i < len(parts):
                if re.match(pattern, parts[i]):
                    # Found a date, this might be the start of a transaction
                    block_start = i
                    # Find the end of this block (next date or end of string)
                    i += 1
                    while i < len(parts) and not re.match(pattern, parts[i]):
                        i += 1
                        
                    # Combine parts into a transaction block
                    block = ''.join(parts[block_start:i])
                    transaction_blocks.append(block)
                else:
                    i += 1
        
        # If we couldn't find transaction blocks with the above methods,
        # try a more generic approach by looking for key fields
        if not transaction_blocks:
            logger.info("No explicit transaction blocks found, trying field-based extraction")
            
            # Split by lines and look for key fields
            lines = ai_response.split('\n')
            current_block = []
            
            for line in lines:
                line = line.strip()
                if line.startswith(('Date:', 'Description:', 'Amount:', 'Type:', 'Category:')) and current_block:
                    # This looks like the start of a new transaction, save the previous block
                    transaction_blocks.append('\n'.join(current_block))
                    current_block = [line]
                elif line.startswith(('Date:', 'Description:', 'Amount:', 'Type:', 'Category:')):
                    # First field of a transaction
                    current_block = [line]
                elif current_block:
                    # Add line to current block
                    current_block.append(line)
            
            # Add the last block if it exists
            if current_block:
                transaction_blocks.append('\n'.join(current_block))
        
        logger.info(f"Found {len(transaction_blocks)} potential transaction blocks")
        
        # Process each transaction block
        for block in transaction_blocks:
            current_tx = {}
            
            # Extract fields from the block
            lines = block.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Try to match key fields with flexible pattern matching
                if line.lower().startswith(('date:', 'date :', 'date -')):
                    field_name = 'date'
                    value = line.split(':', 1)[1].strip() if ':' in line else line.split('-', 1)[1].strip()
                elif line.lower().startswith(('description:', 'description :', 'desc:', 'merchant:', 'payee:')):
                    field_name = 'description'
                    value = line.split(':', 1)[1].strip() if ':' in line else ''
                elif line.lower().startswith(('amount:', 'amount :', 'sum:', 'total:', 'value:')):
                    field_name = 'amount'
                    value = line.split(':', 1)[1].strip() if ':' in line else ''
                elif line.lower().startswith(('type:', 'type :', 'transaction type:')):
                    field_name = 'type'
                    value = line.split(':', 1)[1].strip() if ':' in line else ''
                elif line.lower().startswith(('category:', 'category :', 'cat:')):
                    field_name = 'category'
                    value = line.split(':', 1)[1].strip() if ':' in line else ''
                else:
                    # Not a recognized field header, might be continuation of previous field
                    continue
                    
                # Process the value based on the field type
                if field_name == 'date':
                    if "[UNCLEAR]" in value:
                        # Keep the unclear tag
                        current_tx['date'] = value
                    else:
                        try:
                            # Try different date formats
                            date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d.%m.%Y', '%Y/%m/%d']
                            parsed_date = None
                            
                            for fmt in date_formats:
                                try:
                                    # Extract date part if there's more text
                                    import re
                                    date_match = re.search(r'\d{1,4}[-./]\d{1,2}[-./]\d{1,4}', value)
                                    if date_match:
                                        value = date_match.group(0)
                                        
                                    parsed_date = pd.to_datetime(value, format=fmt, errors='raise')
                                    break
                                except:
                                    continue
                                    
                            if parsed_date is not None:
                                current_tx['date'] = parsed_date.strftime('%Y-%m-%d')
                            else:
                                # If all formats fail, mark as unclear
                                current_tx['date'] = f"[UNCLEAR] {value}"
                        except:
                            # If parsing fails, mark as unclear
                            current_tx['date'] = f"[UNCLEAR] {value}"
                
                elif field_name == 'amount':
                    try:
                        if "[UNCLEAR]" in value:
                            current_tx['amount'] = value
                        else:
                            # Extract numeric part with sign
                            import re
                            
                            # Look for currency symbols and determine sign
                            is_negative = '-' in value or any(neg in value.lower() for neg in ['expense', 'debit', 'payment', 'withdrawal'])
                            is_positive = '+' in value or any(pos in value.lower() for pos in ['income', 'credit', 'deposit', 'received'])
                            
                            # Extract numeric part
                            amount_match = re.search(r'-?\d+[.,]?\d*', value)
                            if amount_match:
                                amount_str = amount_match.group(0).replace(',', '.')
                                amount = float(amount_str)
                                
                                # Apply sign based on context if not explicit in the number
                                if amount > 0 and is_negative and not amount_str.startswith('-'):
                                    amount = -amount
                                elif amount > 0 and not is_positive and not is_negative:
                                    # Default to expense if type is expense
                                    if 'type' in current_tx and current_tx['type'] == 'expense':
                                        amount = -amount
                                        
                                current_tx['amount'] = amount
                            else:
                                current_tx['amount'] = f"[UNCLEAR] {value}"
                    except:
                        current_tx['amount'] = f"[UNCLEAR] {value}"
                
                elif field_name == 'type':
                    # Normalize transaction type
                    value_lower = value.lower()
                    if any(exp in value_lower for exp in ['expense', 'debit', 'payment', 'withdrawal', 'purchase']):
                        current_tx['type'] = 'expense'
                    elif any(inc in value_lower for inc in ['income', 'credit', 'deposit', 'received', 'salary']):
                        current_tx['type'] = 'income'
                    else:
                        # Try to infer from amount if available
                        if 'amount' in current_tx and isinstance(current_tx['amount'], (int, float)):
                            current_tx['type'] = 'income' if current_tx['amount'] > 0 else 'expense'
                        else:
                            # Default to expense if can't determine
                            current_tx['type'] = 'expense'
                
                elif field_name == 'description':
                    current_tx['description'] = value
                
                elif field_name == 'category':
                    current_tx['category'] = value
            
            # Check if we have the minimum required fields for a transaction
            if 'date' in current_tx and ('description' in current_tx or 'amount' in current_tx):
                # Add default values for missing fields
                if 'description' not in current_tx:
                    current_tx['description'] = 'Unknown transaction'
                
                if 'amount' not in current_tx:
                    current_tx['amount'] = '[UNCLEAR] amount'
                
                if 'type' not in current_tx:
                    # Try to infer from amount
                    if isinstance(current_tx['amount'], (int, float)):
                        current_tx['type'] = 'income' if current_tx['amount'] > 0 else 'expense'
                    else:
                        current_tx['type'] = 'expense'  # Default
                
                # Only add if we have valid data
                transactions.append(current_tx)
        
        if not transactions:
            logger.warning("No transactions found in AI response")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(transactions)
        
        # Add default currency if missing
        if 'currency' not in df.columns:
            df['currency'] = self.config.get('currencies', ['KZT'])[0]
        
        # Add default category if missing
        if 'category' not in df.columns:
            df['category'] = df.apply(
                lambda row: self._categorize_transaction(row['description'], row.get('amount'), row.get('type', 'expense')),
                axis=1
            )
        
        # Ensure all required columns exist
        required_columns = ['date', 'description', 'amount', 'currency', 'type', 'category']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Final cleanup - make sure amounts are numeric where possible
        def clean_amount(val):
            if isinstance(val, (int, float)):
                return val
            if isinstance(val, str) and not val.startswith('[UNCLEAR]'):
                try:
                    return float(re.sub(r'[^0-9.-]', '', val))
                except:
                    return val
            return val
        
        df['amount'] = df['amount'].apply(clean_amount)
        
        logger.info(f"Successfully parsed {len(df)} transactions")
        return df[required_columns]
    
    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame to match required format."""
        if df.empty:
            return df
        
        # Find relevant columns
        date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time'])]
        amount_cols = [col for col in df.columns if any(word in col.lower() for word in ['amount', 'sum', 'total'])]
        desc_cols = [col for col in df.columns if any(word in col.lower() for word in ['desc', 'name', 'merchant', 'memo'])]
        
        # Use best matches or defaults
        date_col = date_cols[0] if date_cols else df.columns[0]
        amount_col = amount_cols[0] if amount_cols else (df.columns[1] if len(df.columns) > 1 else None)
        desc_col = desc_cols[0] if desc_cols else (df.columns[2] if len(df.columns) > 2 else None)
        
        # Create standardized DataFrame
        result_df = pd.DataFrame()
        
        # Process date column
        try:
            result_df['date'] = pd.to_datetime(df[date_col], format='%Y-%m-%d', errors='coerce')
        except:
            # If we can't parse dates, use original values with UNCLEAR tag
            result_df['date'] = df[date_col].apply(lambda x: f"[UNCLEAR] {x}")
        
        # Process description column
        if desc_col:
            result_df['description'] = df[desc_col]
        else:
            result_df['description'] = 'Unknown Transaction'
        
        # Process amount column
        if amount_col:
            try:
                result_df['amount'] = pd.to_numeric(df[amount_col], errors='coerce')
            except:
                # If we can't parse amounts, use original values with UNCLEAR tag
                result_df['amount'] = df[amount_col].apply(lambda x: f"[UNCLEAR] {x}")
        else:
            result_df['amount'] = 0
        
        # Determine transaction type based on amount
        try:
            result_df['type'] = result_df['amount'].apply(lambda x: 'income' if float(x) > 0 else 'expense')
        except:
            # Default to expense if we can't determine type
            result_df['type'] = 'expense'
        
        # Add currency from config
        result_df['currency'] = self.config.get('currencies', ['KZT'])[0]
        
        # Categorize transactions
        result_df['category'] = result_df.apply(
            lambda row: self._categorize_transaction(row['description'], row['amount'], row['type']),
            axis=1
        )
        
        # Format date if it's a datetime
        try:
            result_df['date'] = result_df['date'].dt.strftime('%Y-%m-%d')
        except:
            pass  # Keep as is if not datetime
        
        # Handle NaN values
        result_df = result_df.fillna({
            'description': 'Unknown Transaction',
            'category': 'Other',
            'amount': '[UNCLEAR]'
        })
        
        return result_df[['date', 'description', 'amount', 'currency', 'type', 'category']]
    
    def _categorize_transaction(self, description, amount, transaction_type):
        """Categorize transaction based on description and type."""
        if not isinstance(description, str):
            description = str(description)
        
        description_lower = description.lower()
        
        # Get categories from config based on transaction type
        categories_key = f"{transaction_type}_categories"
        if categories_key in self.config:
            categories = [cat["name"] for cat in self.config[categories_key]]
        else:
            # Default categories
            if transaction_type == 'income':
                categories = ["Salary", "Other Income"]
            else:
                categories = ["Food", "Other Expense"]
        
        if not categories:
            return "Other"
        
        # Keyword-based categorization
        category_keywords = {
            "Food": ["grocery", "supermarket", "food", "market", "продукты", "магазин", "еда", "молоко", "магнит"],
            "Transport": ["gas", "fuel", "uber", "taxi", "train", "bus", "transport", "такси", "бензин", "метро"],
            "Utilities": ["electric", "gas", "water", "internet", "phone", "utility", "bill", "счет", "квартплата", "коммунальные"],
            "Shopping": ["amazon", "store", "shop", "mall", "purchase", "одежда", "покупка", "торговый"],
            "Entertainment": ["movie", "cinema", "netflix", "spotify", "game", "entertainment", "кино", "развлечения"],
            "Health": ["pharmacy", "doctor", "hospital", "medical", "health", "аптека", "больница", "врач"],
            "Education": ["school", "university", "course", "book", "education", "школа", "университет", "книга"],
            "Salary": ["salary", "payroll", "wage", "income", "зарплата", "доход", "оклад"],
            "Freelance": ["freelance", "gig", "project", "фриланс", "проект"],
            "Investment": ["dividend", "interest", "investment", "stock", "дивиденды", "инвестиции"],
            "Gift": ["gift", "present", "подарок"]
        }
        
        # Check for keyword matches
        for category, keywords in category_keywords.items():
            if category in categories and any(keyword in description_lower for keyword in keywords):
                return category
        
        # Default to first category if no match
        return categories[0]

    def _process_document_with_openai(self, file_path: str, file_content: bytes) -> pd.DataFrame:
        """Process document using OpenAI's image/document understanding capabilities.
        This is a fallback method when specialized libraries aren't available."""
        logger.info(f"Processing document with OpenAI: {file_path}")
        
        try:
            # Check if it's a PDF
            if file_path.lower().endswith('.pdf'):
                logger.info("Converting PDF to images for OpenAI processing")
                try:
                    # Try to use PyMuPDF if available (even if other enhanced processing libs aren't)
                    import fitz
                    import tempfile
                    from PIL import Image
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        doc = fitz.open(file_path)
                        all_text = ""
                        
                        # Process each page
                        for page_num in range(len(doc)):
                            page = doc.load_page(page_num)
                            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                            image_file = f"{temp_dir}/page_{page_num+1}.png"
                            pix.save(image_file)
                            
                            # Process each page image
                            with open(image_file, "rb") as image_file:
                                page_image_data = base64.b64encode(image_file.read()).decode('utf-8')
                                
                            # Get page text using Vision API
                            page_text = self._extract_text_with_vision(page_image_data, "image/png")
                            all_text += f"Page {page_num+1}:\n{page_text}\n\n"
                    
                    # Extract transactions from all text
                    return self._extract_transactions_with_ai(all_text)
                    
                except ImportError:
                    logger.warning("PyMuPDF not available for PDF conversion. Using basic text extraction.")
                    try:
                        # Try basic PDF text extraction with pdfplumber
                        import pdfplumber
                        
                        with pdfplumber.open(file_path) as pdf:
                            all_text = ""
                            for page in pdf.pages:
                                page_text = page.extract_text()
                                if page_text:
                                    all_text += page_text + "\n\n"
                        
                        # Extract transactions from text
                        return self._extract_transactions_with_ai(all_text)
                    except Exception as pdf_error:
                        logger.error(f"Error extracting text from PDF: {pdf_error}")
                        return pd.DataFrame()
            
            # For images, use Vision API directly
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Convert file to base64 for OpenAI API
                file_data = base64.b64encode(file_content).decode('utf-8')
                
                # Determine MIME type
                mime_type = "image/jpeg"  # Default
                if file_path.lower().endswith('.png'):
                    mime_type = "image/png"
                
                # Extract text from image
                extracted_text = self._extract_text_with_vision(file_data, mime_type)
                
                # Extract transactions from text
                return self._extract_transactions_with_ai(extracted_text)
            
            # For other file types, try to extract as text
            return self._extract_transactions_with_ai("No text could be extracted from this file format.")
            
        except Exception as e:
            logger.error(f"Error processing document with OpenAI: {e}")
            return pd.DataFrame()

    def _extract_text_with_vision(self, base64_image: str, mime_type: str) -> str:
        """Extract text from image using OpenAI Vision API."""
        try:
            # Use GPT-4 Vision to extract text
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use Vision model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a document text extractor. Extract all text from the image maintaining the structure."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Extract ALL text from this document, maintaining the structure. Focus on financial information like dates, amounts, descriptions, and categories."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            # Get the extracted text
            extracted_text = response.choices[0].message.content
            logger.info(f"Successfully extracted {len(extracted_text)} characters of text using Vision API")
            return extracted_text
            
        except Exception as e:
            logger.error(f"Error extracting text with Vision API: {e}")
            logger.error(f"Error processing document with OpenAI: {e}")
            return pd.DataFrame()

    def process_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process multiple files and extract transactions.
        
        Args:
            file_paths: List of paths to files to process
            
        Returns:
            Dict with success count, error count, and extracted transactions
        """
        results = {
            "success_count": 0,
            "error_count": 0,
            "transaction_count": 0,
            "errors": [],
            "transactions": []
        }
        
        for file_path in file_paths:
            logger.info(f"Processing file {file_path}")
            
            # Process file
            result = self.process_file(file_path)
            
            if result["success"]:
                results["success_count"] += 1
                transactions = result["transactions"]
                results["transaction_count"] += len(transactions)
                results["transactions"].extend(transactions)
            else:
                results["error_count"] += 1
                error_msg = f"Error processing {os.path.basename(file_path)}: {result.get('error', 'Unknown error')}"
                results["errors"].append(error_msg)
                logger.error(error_msg)
        
        logger.info(f"Processed {len(file_paths)} files: {results['success_count']} successful, {results['error_count']} errors")
        logger.info(f"Extracted {results['transaction_count']} transactions in total")
        
        return results

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using multiple libraries for best results."""
        text_content = ""
        
        try:
            # Try PyMuPDF (fitz) first if available
            if ENHANCED_PROCESSING:
                try:
                    doc = fitz.open(file_path)
                    for page_num in range(len(doc)):
                        page = doc.load_page(page_num)
                        text_content += page.get_text() + "\n\n"
                    logger.info(f"Extracted text from PDF using PyMuPDF: {len(text_content)} characters")
                except Exception as e:
                    logger.warning(f"Failed to extract text with PyMuPDF: {e}")
            
            # If PyMuPDF failed or is not available, try pdfplumber
            if not text_content.strip():
                try:
                    with pdfplumber.open(file_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_content += page_text + "\n\n"
                    logger.info(f"Extracted text from PDF using pdfplumber: {len(text_content)} characters")
                except Exception as e:
                    logger.warning(f"Failed to extract text with pdfplumber: {e}")
            
            # If text extraction failed or found very little text, it might be a scanned PDF
            # Try to extract tables with tabula if available
            if ENHANCED_PROCESSING and (len(text_content.strip()) < 100 or "tabula" in self.config.get("use_additional_tools", [])):
                try:
                    tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
                    
                    if tables and len(tables) > 0:
                        logger.info(f"Found {len(tables)} tables in PDF")
                        for i, table in enumerate(tables):
                            if not table.empty:
                                table_text = f"\nTable {i+1}:\n{table.to_string()}\n"
                                text_content += table_text
                        logger.info(f"Added table data: {len(text_content)} characters total")
                except Exception as e:
                    logger.warning(f"Failed to extract tables with tabula: {e}")
            
            # If we still don't have good text content, try OCR as a last resort
            if ENHANCED_PROCESSING and len(text_content.strip()) < 100:
                logger.info("PDF appears to be scanned or has little text, attempting OCR")
                try:
                    # Convert PDF to images
                    import tempfile
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        doc = fitz.open(file_path)
                        ocr_text = ""
                        
                        for page_num in range(len(doc)):
                            page = doc.load_page(page_num)
                            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
                            image_file = f"{temp_dir}/page_{page_num+1}.png"
                            pix.save(image_file)
                            
                            # OCR the image
                            page_text = self._extract_text_from_image(image_file)
                            ocr_text += page_text + "\n\n"
                        
                        if ocr_text.strip():
                            text_content = ocr_text
                            logger.info(f"Extracted text from PDF using OCR: {len(text_content)} characters")
                except Exception as e:
                    logger.warning(f"Failed to OCR the PDF: {e}")
            
            return text_content
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def _describe_image(self, file_path: str) -> str:
        """Generate a description of an image for financial document extraction."""
        try:
            # Convert image to base64 for OpenAI API
            with open(file_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Determine MIME type
            mime_type = "image/jpeg"  # Default
            if file_path.lower().endswith('.png'):
                mime_type = "image/png"
            
            # Use OpenAI to describe the image
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Use Vision model
                messages=[
                    {
                        "role": "system",
                        "content": "You are a document analysis assistant. Describe all text and numerical information you see in this image, focusing on financial transactions, dates, amounts, and descriptions. Format the information in plain text, preserving the structure as much as possible."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Please describe all text and financial information visible in this image. Include all dates, amounts, descriptions, and any transaction details you can see. Focus on extracting the raw information, not analyzing it."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            description = response.choices[0].message.content
            logger.info(f"Generated image description: {len(description)} characters")
            return description
        
        except Exception as e:
            logger.error(f"Error describing image: {e}")
            return ""


# Example usage
if __name__ == "__main__":
    # Set up processor
    processor = TransactionProcessor()
    
    # Process a sample file
    sample_file = "path/to/your/sample_statement.csv"
    if os.path.exists(sample_file):
        result = processor.process_file(sample_file)
        print(f"Extracted {len(result['transactions'])} transactions")
        print(result['transactions'])
    else:
        print(f"Sample file not found: {sample_file}") 