"""
Document Processing Tools
Tools for extracting and processing financial document data.
"""
import os
import json
import base64
import io
from datetime import datetime
import re
from PIL import Image
import pdfplumber
import logging
from typing import Dict, List, Optional, Union
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentTools:
    """Tools for document processing and data extraction."""
    
    def __init__(self, api_key=None):
        """Initialize document tools with API key."""
        self.api_key = api_key
        if not self.api_key:
            # Try to get from environment
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4.1-mini"  # Updated to newer model
    
    def read_document(self, file) -> dict:
        """
        Extract raw text and fields from a document (PDF/image).
        
        Args:
            file: The document file object
            
        Returns:
            dict: Raw extracted data with fields and confidence
        """
        try:
            file_extension = file.name.split('.')[-1].lower()
            
            if file_extension in ['jpg', 'jpeg', 'png']:
                return self._process_image(file)
            elif file_extension == 'pdf':
                return self._process_pdf(file)
            else:
                return {"error": f"Unsupported file format: {file_extension}", "success": False}
        except Exception as e:
            logger.error(f"Error reading document: {e}")
            return {"error": str(e), "success": False}
    
    def _process_image(self, file) -> dict:
        """Process image files to extract text."""
        try:
            # Convert image to base64
            image = Image.open(file)
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Use AI model to analyze the receipt
            prompt = """
            Analyze this receipt image and extract the transaction information.
            
            Extract and return the following information in JSON format:
            {
                "date": "YYYY-MM-DD",
                "merchant": "Store/merchant name",
                "amount": "Total amount as a number",
                "items": "Brief description of items/services",
                "confidence": "A score from 0-100 indicating your confidence in the extraction"
            }
            """
            
            # TODO: Replace with actual vision model call
            # For now, we'll simulate a response
            logger.info(f"Processing image file: {file.name}")
            
            # Simulate different quality levels based on filename
            if "blur" in file.name.lower() or "low_quality" in file.name.lower():
                confidence = 45
            elif "receipt" in file.name.lower():
                confidence = 80
            else:
                confidence = 65
            
            # Generate simulated response
            response = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "merchant": self._extract_merchant_from_filename(file.name),
                "amount": "-" + str(round(float(datetime.now().timestamp() % 100) + 10, 2)),
                "items": "Various items from receipt",
                "confidence": confidence,
                "success": True
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"error": str(e), "success": False}
    
    def _process_pdf(self, file) -> dict:
        """Process PDF files to extract text."""
        try:
            # Extract text from PDF
            text_content = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            
            if not text_content.strip():
                return {"error": "No text content extracted from PDF", "success": False}
            
            # Use AI to extract data from text
            prompt = f"""
            Extract financial transaction information from this document text. 
            
            Extract and return the following in JSON format:
            {{
                "transactions": [
                    {{
                        "date": "YYYY-MM-DD",
                        "description": "Transaction description",
                        "amount": "Amount as number (negative for expense, positive for income)",
                        "confidence": "A score from 0-100 indicating your confidence in this transaction extraction"
                    }}
                ],
                "document_type": "bank_statement or receipt",
                "overall_confidence": "A score from 0-100 indicating your overall confidence in the extraction"
            }}
            
            Limit to first 5 transactions if there are many.
            Text content:
            {text_content[:3000]}  # Limit text length
            """
            
            # TODO: Replace with actual AI model call
            # For now, simulate a response
            logger.info(f"Processing PDF file: {file.name}")
            
            # Determine if it's likely a bank statement or receipt based on length
            is_bank_statement = len(text_content) > 500
            doc_type = "bank_statement" if is_bank_statement else "receipt"
            
            # Generate different responses based on document type
            if is_bank_statement:
                # Simulate a bank statement with multiple transactions
                transactions = []
                for i in range(3):
                    tx_date = (datetime.now().timestamp() - i*86400)
                    tx_date_str = datetime.fromtimestamp(tx_date).strftime("%Y-%m-%d")
                    
                    tx = {
                        "date": tx_date_str,
                        "description": self._generate_sample_description(i),
                        "amount": f"{-1 * (float(int(tx_date) % 1000) / 10):.2f}",
                        "confidence": 85
                    }
                    transactions.append(tx)
                
                response = {
                    "transactions": transactions,
                    "document_type": doc_type,
                    "overall_confidence": 75,
                    "success": True
                }
            else:
                # Simulate a receipt with a single transaction
                response = {
                    "transactions": [{
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "description": f"Purchase at {self._extract_merchant_from_filename(file.name)}",
                        "amount": f"{-1 * (float(int(datetime.now().timestamp()) % 100) + 10):.2f}",
                        "confidence": 90
                    }],
                    "document_type": doc_type,
                    "overall_confidence": 90,
                    "success": True
                }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return {"error": str(e), "success": False}
    
    def _extract_merchant_from_filename(self, filename):
        """Extract merchant name from filename for simulation."""
        known_merchants = {
            "magnum": "Magnum Supermarket",
            "small": "Small Market",
            "kaspi": "Kaspi Bank",
            "mechta": "Mechta Electronics",
            "sulpak": "Sulpak",
            "technodom": "Technodom",
            "starbucks": "Starbucks Coffee",
            "kfc": "KFC Restaurant"
        }
        
        filename_lower = filename.lower()
        
        for key, name in known_merchants.items():
            if key in filename_lower:
                return name
        
        return "Unknown Merchant"
    
    def _generate_sample_description(self, index):
        """Generate sample transaction descriptions for simulation."""
        descriptions = [
            "Payment to Magnum Supermarket",
            "Monthly utility bill payment",
            "Salary deposit",
            "Transfer to savings account",
            "Amazon.com purchase",
            "Netflix subscription",
            "Restaurant payment",
            "Gas station purchase"
        ]
        
        return descriptions[index % len(descriptions)]
    
    def generate_summary(self, data):
        """
        Generate a cleaned, structured summary of transaction data.
        
        Args:
            data: Raw transaction data
            
        Returns:
            dict: Structured, cleaned transaction data
        """
        try:
            if not isinstance(data, dict):
                return {"error": "Data must be a dictionary", "success": False}
            
            if "transactions" not in data:
                return {"error": "Missing transactions data", "success": False}
            
            transactions = data.get("transactions", [])
            cleaned_transactions = []
            
            for tx in transactions:
                # Skip transactions with no date or amount
                if not tx.get("date") or not tx.get("amount"):
                    continue
                
                # Convert amount to float
                try:
                    amount_float = float(tx.get("amount"))
                except:
                    # Skip if amount can't be converted to float
                    continue
                
                # Determine transaction type
                tx_type = "expense" if amount_float < 0 else "income"
                
                # Get description
                description = tx.get("description", "Unknown transaction")
                
                # Get category if available or set to Unknown
                category = tx.get("category", "Unknown")
                
                cleaned_tx = {
                    "date": tx.get("date"),
                    "description": description,
                    "amount": abs(amount_float),  # Store as positive
                    "type": tx_type,
                    "category": category,
                    "currency": tx.get("currency", "KZT")
                }
                
                cleaned_transactions.append(cleaned_tx)
            
            return {
                "success": True,
                "transactions": cleaned_transactions,
                "total_count": len(cleaned_transactions),
                "income_count": sum(1 for tx in cleaned_transactions if tx["type"] == "income"),
                "expense_count": sum(1 for tx in cleaned_transactions if tx["type"] == "expense"),
                "summary_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"error": str(e), "success": False} 