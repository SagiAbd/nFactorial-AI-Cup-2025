"""
Document Processing Agent
Uses AI and tools to process financial documents and extract transactions.
"""
import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pandas as pd
from openai import OpenAI

from agents.tools import DocumentTools
from core.data_manager import load_transactions, save_transaction, load_config, TRANSACTIONS_FILE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentAgent:
    """Agent for processing financial documents and extracting transactions."""
    
    def __init__(self, api_key=None):
        """Initialize the document agent with API key."""
        self.api_key = api_key
        if not self.api_key:
            # Try to get from environment
            self.api_key = os.getenv('OPENAI_API_KEY')
            if not self.api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-4.1-mini"  # Updated to newer model
        
        # Initialize tools
        self.tools = DocumentTools(api_key=self.api_key)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize state
        self.current_document = None
        self.extracted_data = None
        self.transaction_data = None
        self.chat_history = []
    
    def _load_config(self):
        """Load configuration for the document agent."""
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
    
    def process_document(self, file) -> Dict:
        """
        Process a financial document and extract transactions.
        
        Args:
            file: The document file to process
            
        Returns:
            dict: The extraction results
        """
        try:
            # Store current document
            self.current_document = file
            
            # Extract initial data from document
            logger.info(f"Processing document: {file.name}")
            self.extracted_data = self.tools.read_document(file)
            
            if not self.extracted_data.get("success", False):
                error_msg = self.extracted_data.get("error", "Unknown error")
                logger.error(f"Error extracting data: {error_msg}")
                return {"success": False, "error": error_msg}
            
            # Generate structured data
            self.transaction_data = self.tools.generate_summary(self.extracted_data)
            
            if not self.transaction_data.get("success", False):
                error_msg = self.transaction_data.get("error", "Error generating transaction summary")
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Add record of this processing to chat history
            self._add_to_chat_history("system", f"Processed document: {file.name}", "document_processed")
            
            # Return structured data
            return {
                "success": True,
                "transactions": self.transaction_data.get("transactions", []),
                "document_type": self.extracted_data.get("document_type", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {"success": False, "error": str(e)}
    
    def save_transactions(self):
        """
        Save processed transactions to the database.
        
        Returns:
            dict: The status of the operation
        """
        if not self.transaction_data or not self.transaction_data.get("transactions"):
            return {"success": False, "error": "No transactions to save"}
        
        transactions = self.transaction_data.get("transactions", [])
        
        success_count = 0
        error_count = 0
        
        for tx in transactions:
            try:
                # Convert date string to datetime
                transaction_date = datetime.strptime(tx['date'], '%Y-%m-%d')
                
                # Determine amount (use abs value since type already indicates sign)
                amount = abs(float(tx['amount']))
                
                # Debug information for each transaction
                logger.info(f"Saving: {tx['date']} | {tx['description']} | {amount} {tx['currency']} | {tx['category']} | {tx['type']}")
                
                # Save the transaction
                result = save_transaction(
                    transaction_date=transaction_date,
                    amount=amount,
                    category=tx['category'],
                    currency=tx['currency'],
                    transaction_type=tx['type'],
                    description=tx['description']
                )
                
                if result:
                    logger.info(f"Successfully saved transaction")
                    success_count += 1
                else:
                    logger.error(f"Failed to save transaction")
                    error_count += 1
            except Exception as e:
                logger.error(f"Error saving transaction: {str(e)}")
                error_count += 1
        
        # Add to chat history
        self._add_to_chat_history(
            "system",
            f"Saved {success_count} transactions, {error_count} errors",
            "save_transactions"
        )
        
        return {
            "success": True,
            "saved_count": success_count,
            "error_count": error_count,
            "total_count": len(transactions)
        }
    
    def get_chat_completions(self, user_input):
        """
        Get chat completions from the AI model.
        
        Args:
            user_input: The user's message
            
        Returns:
            dict: The AI's response
        """
        try:
            # Add user message to history
            self._add_to_chat_history("user", user_input, "message")
            
            # Create messages for the AI
            messages = self._prepare_messages_for_ai()
            
            # Call the AI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract the assistant's message
            assistant_message = response.choices[0].message.content
            
            # Add to chat history
            self._add_to_chat_history("assistant", assistant_message, "message")
            
            return {"success": True, "message": assistant_message}
            
        except Exception as e:
            logger.error(f"Error getting chat completions: {e}")
            return {"success": False, "error": str(e)}
    
    def _prepare_messages_for_ai(self):
        """Prepare messages for the AI model."""
        # Start with a system message
        messages = [{
            "role": "system",
            "content": f"""You are a financial document assistant helping with transaction processing.
            You can help explain transactions, suggest categories, and provide insights.
            Current document status:
            - Processed document: {self.current_document.name if self.current_document else 'None'}
            - Transactions found: {len(self.transaction_data.get('transactions', [])) if self.transaction_data else 0}
            """
        }]
        
        # Add transaction summary if available
        if self.transaction_data and self.transaction_data.get("transactions"):
            tx_count = len(self.transaction_data["transactions"])
            income_count = self.transaction_data.get("income_count", 0)
            expense_count = self.transaction_data.get("expense_count", 0)
            
            messages.append({
                "role": "system",
                "content": f"""Document summary:
                - Total transactions: {tx_count}
                - Income transactions: {income_count}
                - Expense transactions: {expense_count}
                
                You can discuss these transactions and help the user understand them.
                """
            })
        
        # Add chat history (last 10 messages)
        for entry in self.chat_history[-10:]:
            if entry["type"] == "message":
                messages.append({
                    "role": entry["role"],
                    "content": entry["content"]
                })
        
        return messages
    
    def _add_to_chat_history(self, role, content, entry_type):
        """Add an entry to the chat history."""
        self.chat_history.append({
            "role": role,
            "content": content,
            "type": entry_type,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    
    def get_transaction_summary(self):
        """Get summary of transactions in the current document."""
        if not self.transaction_data or not self.transaction_data.get("transactions"):
            return {"success": False, "error": "No transactions available"}
        
        return {
            "success": True,
            "total": self.transaction_data.get("total_count", 0),
            "income_count": self.transaction_data.get("income_count", 0),
            "expense_count": self.transaction_data.get("expense_count", 0),
            "transactions": self.transaction_data.get("transactions", [])
        }
    
    def categorize_transactions(self):
        """Automatically categorize transactions using AI."""
        if not self.transaction_data or not self.transaction_data.get("transactions"):
            return {"success": False, "error": "No transactions to categorize"}
        
        transactions = self.transaction_data.get("transactions", [])
        
        # Get available categories
        income_cats = [cat['name'] for cat in self.config['income_categories']]
        expense_cats = [cat['name'] for cat in self.config['expense_categories']]
        
        # Create a simple description of each transaction
        tx_list = []
        for i, tx in enumerate(transactions):
            tx_list.append(f"{i}: {tx.get('date', 'unknown')} | {tx.get('description', 'unknown')} | {tx.get('amount', '0')} | {tx.get('type', 'unknown')}")
        
        # Join into a single string
        transactions_text = "\n".join(tx_list)
        
        # Create AI prompt
        prompt = f"""
        Categorize these financial transactions:
        
        Available Income Categories: {', '.join(income_cats)}
        Available Expense Categories: {', '.join(expense_cats)}
        
        Transactions:
        {transactions_text}
        
        For each transaction, respond with:
        INDEX: SUGGESTED_CATEGORY
        
        Only suggest categories from the available lists above.
        Make sure to use income categories for income transactions and expense categories for expense transactions.
        """
        
        try:
            # Call the AI
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=800
            )
            
            # Extract the AI's suggestions
            ai_response = response.choices[0].message.content
            
            # Parse the suggestions
            updated_count = 0
            for line in ai_response.split('\n'):
                if ':' in line:
                    try:
                        parts = line.split(':', 1)
                        idx = int(parts[0].strip())
                        category = parts[1].strip()
                        
                        # Validate category exists
                        tx_type = transactions[idx].get('type', 'expense')
                        valid_categories = income_cats if tx_type == 'income' else expense_cats
                        
                        if category in valid_categories and idx < len(transactions):
                            transactions[idx]['category'] = category
                            updated_count += 1
                    except:
                        continue
            
            # Update transaction data
            self.transaction_data["transactions"] = transactions
            
            # Add to chat history
            self._add_to_chat_history(
                "system",
                f"Auto-categorized {updated_count} transactions",
                "categorization"
            )
            
            return {
                "success": True,
                "updated_count": updated_count,
                "total_count": len(transactions)
            }
            
        except Exception as e:
            logger.error(f"Error categorizing transactions: {e}")
            return {"success": False, "error": str(e)} 