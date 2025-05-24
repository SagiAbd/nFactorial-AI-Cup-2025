"""
LangChain-powered AI agents for financial assistance and insights.
"""
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI model
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

class InsightsAgent:
    """Agent for generating financial insights from transaction data."""
    
    def __init__(self):
        self.prompt = PromptTemplate(
            input_variables=["transaction_data"],
            template="""
            Based on the following financial transaction data, provide 3-5 insightful observations about spending patterns, saving opportunities, or financial behaviors.
            
            Transaction Data:
            {transaction_data}
            
            Provide your insights in a clear, bulleted format.
            """
        )
        self.chain = LLMChain(llm=llm, prompt=self.prompt)
    
    def generate_insights(self):
        """Generate financial insights based on transaction data."""
        # In a real implementation, you would fetch actual transaction data
        # For now, return a placeholder message
        return "• Your spending in the Food category has increased by 15% this month.\n• You've been consistent with your savings goals, great job!\n• Consider reviewing your subscription services, as they account for 20% of your monthly expenses."


class GoalProgressAgent:
    """Agent for tracking progress toward financial goals."""
    
    def __init__(self):
        self.prompt = PromptTemplate(
            input_variables=["goals", "transactions"],
            template="""
            Based on the following financial goals and recent transactions, provide an assessment of progress toward these goals.
            
            Goals:
            {goals}
            
            Recent Transactions:
            {transactions}
            
            Provide a concise progress report with specific metrics and suggestions.
            """
        )
        self.chain = LLMChain(llm=llm, prompt=self.prompt)
    
    def check_progress(self):
        """Check progress toward financial goals."""
        # Placeholder implementation
        return "• Savings Goal: 65% complete (₸325,000 of ₸500,000)\n• Debt Reduction: On track, 40% of target achieved\n• Vacation Fund: Behind schedule, consider allocating an extra ₸10,000 this month"
    
    def check_spending_warning(self, category, amount):
        """Check if a new expense might affect goal progress."""
        # Simple implementation that warns about large expenses
        if amount > 50000 and category in ["Entertainment", "Shopping", "Dining"]:
            return f"This {category} expense of ₸{amount} might impact your monthly savings goal."
        return None


class ChatAssistantAgent:
    """Agent for conversational financial assistance."""
    
    def __init__(self):
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "user_question"],
            template="""
            You are a helpful financial assistant. Based on the following chat history and the user's question, provide a helpful, concise response.
            
            Chat History:
            {chat_history}
            
            User Question:
            {user_question}
            
            Your response should be friendly, practical, and focused on helping the user with their financial concerns.
            """
        )
        self.chain = LLMChain(llm=llm, prompt=self.prompt)
    
    def generate_response(self, user_input):
        """Generate a response to user input."""
        # In a real implementation, you would include chat history
        # Placeholder implementation for now
        return f"I'd be happy to help with your question about '{user_input}'. To provide more specific advice, I would need to analyze your transaction history and financial goals. Is there a particular aspect of your finances you'd like me to focus on?" 