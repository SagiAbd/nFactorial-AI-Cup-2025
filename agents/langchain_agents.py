import os
import json
import pandas as pd
from PIL import Image
import pytesseract
from pathlib import Path
from difflib import get_close_matches
from datetime import datetime
import re

# Try to import LangChain components with fallback
try:
    from langchain_openai import OpenAI
    from langchain.agents import Tool, initialize_agent, AgentType
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationChain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain not available. Using simple fallback agent.")

DATA_DIR = Path("data")
CLAR_PATH = DATA_DIR / "clarifications.json"
CAT_PATH = DATA_DIR / "categories.json"
TRANS_PATH = DATA_DIR / "transactions.csv"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True, parents=True)

# — Utilities for clarifications & categories —

def load_json(path, default):
    """Load JSON file with default fallback"""
    path.parent.mkdir(exist_ok=True, parents=True)
    if not path.exists():
        path.write_text(json.dumps(default, indent=2))
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default

def save_json(path, obj):
    """Save object as JSON"""
    try:
        path.write_text(json.dumps(obj, indent=2))
        return True
    except Exception as e:
        print(f"Error saving JSON: {e}")
        return False

def check_clarification(text, field_type):
    """Check if we have a clarification for unclear text"""
    try:
        clar = load_json(CLAR_PATH, [])
        for e in clar:
            if (e.get("field_type") == field_type and 
                get_close_matches(text, [e.get("unclear_text", "")], cutoff=0.8)):
                return e.get("clarified_value")
    except Exception as e:
        print(f"Error checking clarification: {e}")
    return None

def add_clarification(unclear, clarified, field_type):
    """Add a new clarification mapping"""
    try:
        db = load_json(CLAR_PATH, [])
        db.append({
            "unclear_text": unclear,
            "clarified_value": clarified,
            "field_type": field_type,
            "date_added": datetime.now().isoformat()
        })
        if save_json(CLAR_PATH, db):
            return "✅ Clarification saved successfully"
        else:
            return "❌ Failed to save clarification"
    except Exception as e:
        return f"❌ Error saving clarification: {str(e)}"

def suggest_category(desc):
    """Suggest categories based on description"""
    try:
        cats = load_json(CAT_PATH, {
            "Food & Dining": ["restaurant", "grocery", "food", "dining"],
            "Transportation": ["gas", "uber", "taxi", "transport"],
            "Shopping": ["amazon", "store", "retail", "shopping"],
            "Bills & Utilities": ["electric", "water", "phone", "internet"],
            "Entertainment": ["movie", "game", "entertainment", "fun"]
        })
        
        # Find best matching category
        desc_lower = desc.lower()
        for category, keywords in cats.items():
            if any(keyword in desc_lower for keyword in keywords):
                return [category]
        
        # Fallback to fuzzy matching
        return get_close_matches(desc, list(cats.keys()), n=3, cutoff=0.6)
    except Exception as e:
        print(f"Error suggesting category: {e}")
        return ["Uncategorized"]

def read_file(path: str):
    """Read various file types"""
    try:
        p = Path(path)
        if not p.exists():
            return f"File not found: {path}"
            
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
            return df.to_dict("records")
        elif p.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(p)
            return df.to_dict("records")
        elif p.suffix.lower() in [".jpg", ".jpeg", ".png", ".pdf"]:
            try:
                txt = pytesseract.image_to_string(Image.open(p))
                return txt
            except Exception as ocr_error:
                return f"OCR failed: {str(ocr_error)}"
        else:
            return f"Unsupported file type: {p.suffix}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def extract_fields(raw_text: str):
    """Extract transaction fields from raw text"""
    try:
        # Look for date patterns
        date_match = re.search(r'\b\d{4}-\d{2}-\d{2}\b|\b\d{2}/\d{2}/\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', raw_text)
        
        # Look for amount patterns
        amount_match = re.search(r'-?\$?\d+\.?\d{0,2}', raw_text)
        
        # Look for description (longest alphabetic sequence)
        desc_matches = re.findall(r'[A-Za-z][\w ]{2,}', raw_text)
        description = max(desc_matches, key=len) if desc_matches else "[UNCLEAR]"
        
        # Extract amount value
        amount = None
        if amount_match:
            amount_str = amount_match.group(0).replace('$', '')
            try:
                amount = float(amount_str)
            except ValueError:
                amount = None
        
        return {
            "date": date_match.group(0) if date_match else "[UNCLEAR]",
            "description": description.strip(),
            "amount": amount,
            "type": "expense" if amount and amount < 0 else "income",
            "category": suggest_category(description)[0] if description != "[UNCLEAR]" else "[UNCLEAR]",
            "currency": "USD"
        }
    except Exception as e:
        return {
            "error": f"Failed to extract fields: {str(e)}",
            "raw_text": raw_text
        }

def update_transactions(record: dict):
    """Update transactions CSV with new record"""
    try:
        # Create default CSV if it doesn't exist
        if not TRANS_PATH.exists():
            df = pd.DataFrame(columns=['date', 'description', 'amount', 'type', 'category', 'currency'])
            df.to_csv(TRANS_PATH, index=False)
        
        # Load existing transactions
        df = pd.read_csv(TRANS_PATH)
        
        # Add new record
        new_df = pd.DataFrame([record])
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Save updated transactions
        df.to_csv(TRANS_PATH, index=False)
        return "✅ Transaction saved successfully"
    except Exception as e:
        return f"❌ Error saving transaction: {str(e)}"

# — Define tools for the agent —

TOOLS = [
    Tool(
        name="read_file",
        func=read_file,
        description="Load CSV/XLSX files or perform OCR on PDF/JPG files. Input: file_path (string)"
    ),
    Tool(
        name="extract_fields",
        func=extract_fields,
        description="Parse transaction fields (date, description, amount, type) from raw text. Input: raw_text (string)"
    ),
    Tool(
        name="check_clarification",
        func=check_clarification,
        description="Look up past clarifications for unclear text. Input: text (string), field_type (string)"
    ),
    Tool(
        name="add_clarification",
        func=add_clarification,
        description="Save a new unclear→clarified text mapping. Input: unclear_text, clarified_value, field_type"
    ),
    Tool(
        name="suggest_category",
        func=suggest_category,
        description="Suggest transaction categories based on description. Input: description (string)"
    ),
    Tool(
        name="update_transactions",
        func=update_transactions,
        description="Save a cleaned transaction record to CSV. Input: record (dict with date, description, amount, type, category)"
    ),
]

class SimpleChatAgent:
    """Simple fallback agent when LangChain is not available"""
    
    def __init__(self):
        self.conversation_history = []
    
    def generate_response(self, prompt: str, context: dict = None) -> str:
        """Generate a simple response based on keywords and patterns"""
        # Save conversation for context
        self.conversation_history.append({"role": "user", "content": prompt})
        
        prompt_lower = prompt.lower()
        
        # Handle financial analysis requests
        if any(word in prompt_lower for word in ['spending', 'budget', 'summary', 'analysis']):
            response = "I'd be happy to help with financial analysis! Please ensure your transaction data is loaded first."
        
        # Handle category requests
        elif any(word in prompt_lower for word in ['category', 'categories', 'categorize']):
            response = "I can help categorize your transactions! Upload your transaction data and I'll analyze your spending by category."
        
        # Handle file processing requests
        elif any(word in prompt_lower for word in ['upload', 'file', 'receipt', 'process']):
            response = "I can help process financial files! Here's what I can do:\n• 📄 Read CSV/Excel transaction files\n• 🧾 Extract data from receipt images using OCR\n• 🏷️ Automatically categorize transactions\n• 💾 Save processed data to your transaction history"
        
        # Handle goal/planning requests
        elif any(word in prompt_lower for word in ['goal', 'plan', 'save', 'budget']):
            response = "I can help with financial planning! Based on your spending patterns, I can:\n• 🎯 Suggest realistic savings goals\n• 📊 Identify areas to reduce spending\n• 📈 Track your progress over time\n• 💡 Provide personalized recommendations"
        
        # Check for greetings
        elif any(word in prompt_lower for word in ['hi', 'hello', 'hey']):
            response = "👋 Hello! How can I help with your finances today?"
        
        # Check for conversation context/history
        elif any(word in prompt_lower for word in ['previous', 'before', 'last time', 'remember']):
            if len(self.conversation_history) > 2:
                previous = self.conversation_history[-3]['content'] if len(self.conversation_history) >= 3 else "Nothing yet"
                response = f"Yes, I remember our conversation. You previously mentioned: '{previous}'"
            else:
                response = "We're just getting started with our conversation. How can I help you today?"
        
        # Default helpful response
        else:
            response = "I'm here to help with your finances! I can:\n• 📊 Analyze your spending patterns\n• 🏷️ Categorize transactions\n• 📄 Process receipts and financial documents\n• 🎯 Help with budgeting and goal setting\n• 💡 Provide financial insights\n\nWhat would you like to explore?"
        
        # Save response to history
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

class ChatAssistantAgent:
    """Chat assistant agent with LangChain integration or simple fallback"""
    
    def __init__(self):
        if LANGCHAIN_AVAILABLE and os.getenv('OPENAI_API_KEY'):
            try:
                # Initialize conversation memory
                self.memory = ConversationBufferMemory(return_messages=True)
                
                # Initialize language model
                self.llm = OpenAI(temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))
                
                # Create conversation chain for memory
                self.conversation = ConversationChain(
                    llm=self.llm,
                    memory=self.memory,
                    verbose=False
                )
                
                self.use_langchain = True
                print("✅ LangChain conversation agent initialized successfully")
            except Exception as e:
                print(f"⚠️ LangChain initialization failed: {e}")
                self.use_langchain = False
                self.simple_agent = SimpleChatAgent()
        else:
            self.use_langchain = False
            self.simple_agent = SimpleChatAgent()
            if not LANGCHAIN_AVAILABLE:
                print("ℹ️ Using simple chat agent (LangChain not available)")
            else:
                print("ℹ️ Using simple chat agent (OpenAI API key not found)")
    
    def generate_response(self, prompt: str, context: dict = None) -> str:
        """Generate response using LangChain conversation chain or simple fallback"""
        try:
            if self.use_langchain:
                # Add context to prompt if available
                enhanced_prompt = prompt
                if context:
                    context_str = "\n".join([f"{k}: {v}" for k, v in context.items() if k != "transactions"])
                    enhanced_prompt = f"[Context: {context_str}]\n\nUser: {prompt}"
                
                # Use the conversation chain to maintain memory
                response = self.conversation.predict(input=enhanced_prompt)
                return response
            else:
                return self.simple_agent.generate_response(prompt, context)
                
        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}"
            
            # Fallback to simple agent if LangChain fails
            if self.use_langchain:
                try:
                    if not hasattr(self, 'simple_agent'):
                        self.simple_agent = SimpleChatAgent()
                    return self.simple_agent.generate_response(prompt, context)
                except Exception as fallback_error:
                    return f"{error_msg}\n\nFallback also failed: {str(fallback_error)}"
            
            return error_msg