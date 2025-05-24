# Architecture: PDF/CSV/JPG/XLSX reader -> Data Store -> Clarification DB -> Tools
#                -> LangChain Agent w/ Memory -> Chat Interface (Streamlit)

from pathlib import Path
import json
import pandas as pd
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from core.data_manager import load_transactions, get_financial_summary, get_category_spending

# --- Config & Paths --------------------------------
DATA_DIR = Path("data")
CLAR_DIR = DATA_DIR / "clarifications.json"
CAT_DIR = DATA_DIR / "categories.json"
TRANS_DIR = DATA_DIR / "transactions.csv"

# Ensure files exist
DATA_DIR.mkdir(exist_ok=True)
CLAR_DIR.touch(exist_ok=True)
CAT_DIR.touch(exist_ok=True)
CLAR_DIR.write_text(CLAR_DIR.read_text() if CLAR_DIR.exists() and CLAR_DIR.read_text() else "[]")
CAT_DIR.write_text(CAT_DIR.read_text() if CAT_DIR.exists() and CAT_DIR.read_text() else "{}")

# --- Utility Functions -----------------------------
def save_transactions(df):
    df.to_csv(TRANS_DIR, index=False)

# Clarification DB
def load_clarifications():
    return json.loads(CLAR_DIR.read_text())

def save_clarifications(clist):
    CLAR_DIR.write_text(json.dumps(clist, indent=2))

# Category DB
def load_categories():
    return json.loads(CAT_DIR.read_text())

def save_categories(cdict):
    CAT_DIR.write_text(json.dumps(cdict, indent=2))

# --- Tools -----------------------------------------
def read_file(path: str):
    path = Path(path)
    if path.suffix.lower() in ['.csv', '.xlsx']:
        df = pd.read_csv(path) if path.suffix=='.csv' else pd.read_excel(path)
        return df.to_dict(orient='records')
    if path.suffix.lower() in ['.pdf', '.jpg', '.png']:
        text = pytesseract.image_to_string(Image.open(path))
        return text
    return f"Unsupported file type: {path.suffix}"  

# Field extractor: naive regex or pandas inference
import re

def extract_fields(raw_text: str):
    # Simplified: find date, amount, description
    date = re.search(r"\d{4}-\d{2}-\d{2}", raw_text)
    amt = re.search(r"-?\d+\.\d{2}", raw_text)
    desc = re.search(r"[A-Z][\w ]{2,}", raw_text)
    return {
        'date': date.group(0) if date else '[UNCLEAR]',
        'description': desc.group(0) if desc else '[UNCLEAR]',
        'amount': float(amt.group(0)) if amt else None,
        'type': 'expense' if amt and float(amt)<0 else 'income',
        'category': '[UNCLEAR]',
        'currency': 'USD'
    }

# Clarification lookup
from difflib import get_close_matches

def check_clarification(text, field_type):
    clar = load_clarifications()
    for entry in clar:
        if entry['field_type']==field_type and get_close_matches(text, [entry['unclear_text']], cutoff=0.8):
            return entry['clarified_value']
    return None

def add_clarification(unclear, clarified, ftype):
    db = load_clarifications()
    db.append({
        'unclear_text': unclear,
        'clarified_value': clarified,
        'field_type': ftype,
        'date_added': pd.Timestamp.now().isoformat()
    })
    save_clarifications(db)
    return "Saved."

# Category suggestion
def suggest_category(desc):
    cats = load_categories()
    matches = get_close_matches(desc, list(cats.keys()), cutoff=0.6)
    return matches[:3]

# Graphs & insights tool
def create_spending_chart():
    df = load_transactions()
    df = df[df['transaction_type']=='expense']
    by_cat = df.groupby('category')['amount'].sum().abs()
    plt.figure()
    by_cat.plot(kind='bar')
    plt.tight_layout()
    plt.savefig('data/spending.png')
    return 'data/spending.png'

# --- Define Tools for Agent -----------------------
tools = [
    Tool(
        name="read_file",
        func=read_file,
        description="Read CSV, XLSX, PDF, JPG and return raw data or text."
    ),
    Tool(
        name="extract_fields",
        func=extract_fields,
        description="Extract date, description, amount, type, category from raw text."
    ),
    Tool(
        name="check_clarification",
        func=check_clarification,
        description="Check clarifications.json for a match."
    ),
    Tool(
        name="add_clarification",
        func=add_clarification,
        description="Add a new clarification to clarifications.json."
    ),
    Tool(
        name="suggest_category",
        func=suggest_category,
        description="Suggest up to 3 categories for a given description."
    ),
    Tool(
        name="create_spending_chart",
        func=create_spending_chart,
        description="Generate and save a spending by category bar chart."
    ),
    Tool(
        name="get_transactions",
        func=load_transactions,
        description="Load all transactions from the database."
    ),
    Tool(
        name="get_financial_summary",
        func=get_financial_summary,
        description="Get a summary of financial data including income, expenses, and balance."
    ),
    Tool(
        name="get_category_spending",
        func=get_category_spending,
        description="Get spending breakdown by category."
    )
]

# --- Agent Initialization -------------------------
memory = ConversationBufferMemory(memory_key="chat_history")
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools,                # your suite of read/extract/clarify/chart tools
    llm,                  # the underlying LLM
    agent="chat-zero-shot-react-description",
    verbose=True,
    memory=memory
)

# --- Agent Classes --------------------------------
class ChatAssistantAgent:
    """Chat-based financial assistance"""
    
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.llm = OpenAI(temperature=0.2)
        self.agent = initialize_agent(
            tools, 
            self.llm,
            agent="chat-zero-shot-react-description",
            verbose=True,
            memory=self.memory
        )
    
    def generate_response(self, user_input):
        """Generate a response to the user's input"""
        try:
            response = self.agent.run(user_input)
            return response
        except Exception as e:
            return f"I encountered an error: {str(e)}. Please try a different question."


class InsightsAgent:
    """Financial insights generator"""
    
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.llm = OpenAI(temperature=0.2)
        self.agent = initialize_agent(
            tools, 
            self.llm,
            agent="chat-zero-shot-react-description",
            verbose=True,
            memory=self.memory
        )
    
    def generate_insights(self):
        """Generate financial insights based on transaction data"""
        try:
            response = self.agent.run(
                "Analyze my financial data and provide useful insights about my spending patterns, "
                "income trends, and actionable recommendations."
            )
            return response
        except Exception as e:
            return f"Unable to generate insights: {str(e)}. Please try again later."


class GoalProgressAgent:
    """Financial goal tracking"""
    
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.llm = OpenAI(temperature=0.2)
        self.agent = initialize_agent(
            tools, 
            self.llm,
            agent="chat-zero-shot-react-description",
            verbose=True,
            memory=self.memory
        )
    
    def check_progress(self):
        """Check progress on financial goals"""
        try:
            response = self.agent.run(
                "Analyze my financial data and check my progress toward financial goals. "
                "Provide updates and recommendations to help me stay on track."
            )
            return response
        except Exception as e:
            return f"Unable to check goal progress: {str(e)}. Please try again later."
    
    def check_spending_warning(self, category, amount):
        """Check if a new expense might exceed category budget"""
        try:
            prompt = f"Check if adding ${amount} to {category} would exceed my budget for this category."
            response = self.agent.run(prompt)
            return response
        except:
            return None


if __name__ == '__main__':
    # Example chat loop
    print("Financial Assistant Agent is running. Type 'exit' to quit.")
    chat_agent = ChatAssistantAgent()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit': break
        response = chat_agent.generate_response(user_input)
        print(f"Agent: {response}")






