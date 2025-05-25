import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import json
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
import requests

# Path to transactions file
TX_PATH = "data/transactions.csv"

# Default exchange rate if API fails
DEFAULT_USD_TO_KZT = 520.0

class CurrencyConverter:
    """Helper class for currency conversion"""
    
    @staticmethod
    def get_exchange_rate(from_currency="USD", to_currency="KZT"):
        """Get the latest exchange rate from API or use default"""
        try:
            # Try to get real-time exchange rate
            url = f"https://open.er-api.com/v6/latest/{from_currency}"
            response = requests.get(url)
            data = response.json()
            
            if data.get("result") == "success" and to_currency in data.get("rates", {}):
                return data["rates"][to_currency]
            return DEFAULT_USD_TO_KZT
        except Exception as e:
            print(f"Failed to get exchange rate: {e}")
            return DEFAULT_USD_TO_KZT

class TransactionAnalyzer:
    """Tool for making intelligent assumptions about transaction patterns"""
    
    def __init__(self, csv_path: str = TX_PATH):
        self.csv_path = csv_path
        self.df = None
        self.exchange_rate = CurrencyConverter.get_exchange_rate()
        self._load_data()
    
    def _create_sample_transactions(self) -> None:
        """Creates a comprehensive sample CSV file"""
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Create more comprehensive sample data
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        np.random.seed(42)  # For reproducible sample data
        
        categories = ['groceries', 'dining', 'transport', 'shopping', 'utilities', 'entertainment', 'healthcare']
        descriptions = [
            'Walmart Grocery', 'Target', 'Safeway', 'Whole Foods',
            'McDonald\'s', 'Starbucks', 'Pizza Hut', 'Local Restaurant',
            'Gas Station', 'Uber', 'Metro Card', 'Parking',
            'Amazon', 'Best Buy', 'Clothing Store', 'Pharmacy',
            'Electric Bill', 'Water Bill', 'Internet Bill',
            'Movie Theater', 'Spotify', 'Gym Membership',
            'Doctor Visit', 'Pharmacy'
        ]
        
        # Include both USD and KZT currencies
        currencies = ['USD', 'KZT']
        
        sample_data = []
        for i in range(50):
            category = np.random.choice(categories)
            desc = np.random.choice([d for d in descriptions if any(cat in d.lower() for cat in [category[:4]])] or descriptions[:4])
            amount = round(np.random.lognormal(3, 1), 2)  # Log-normal distribution for realistic amounts
            
            # Randomly assign USD or KZT (80% USD, 20% KZT for sample diversity)
            currency = np.random.choice(currencies, p=[0.8, 0.2])
            
            # If currency is KZT, adjust the amount to be realistic in KZT (approximately)
            if currency == 'KZT':
                # Use a rough exchange rate for sample data
                amount = amount * self.exchange_rate
                
            sample_data.append({
                'date': dates[i].strftime('%Y-%m-%d'),
                'description': desc,
                'amount': amount,
                'type': 'expense',
                'category': category,
                'currency': currency
            })
        
        df = pd.DataFrame(sample_data)
        df.to_csv(self.csv_path, index=False)
        print(f"Created comprehensive sample transactions file at {self.csv_path}")
    
    def _load_data(self):
        """Load and prepare transaction data"""
        try:
            # Check if file exists and has content
            if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
                print(f"Transaction file {self.csv_path} doesn't exist or is empty. Creating sample data.")
                self._create_sample_transactions()
                
            # Try to read the file
            self.df = pd.read_csv(self.csv_path)
            
            # If the dataframe is empty, create sample data
            if self.df.empty:
                print("Transaction file exists but contains no data. Creating sample data.")
                self._create_sample_transactions()
                self.df = pd.read_csv(self.csv_path)
                
            # Convert date column to datetime
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
            # Ensure amount is numeric
            if 'amount' in self.df.columns:
                self.df['amount'] = pd.to_numeric(self.df['amount'], errors='coerce')
                
            # Convert USD to KZT if currency column exists
            if 'currency' in self.df.columns:
                # Add original_amount column to preserve original values
                self.df['original_amount'] = self.df['amount']
                self.df['original_currency'] = self.df['currency']
                
                # Convert all amounts to KZT
                for idx, row in self.df.iterrows():
                    if row['currency'] == 'USD':
                        self.df.at[idx, 'amount'] = row['amount'] * self.exchange_rate
                
                # Update all currencies to KZT
                self.df['currency'] = 'KZT'
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create an empty dataframe with the expected columns
            self.df = pd.DataFrame(columns=[
                'date', 'description', 'amount', 'type', 'category', 'currency',
                'original_amount', 'original_currency'
            ])
            # Create sample data
            self._create_sample_transactions()
            # Try to load again
            try:
                self.df = pd.read_csv(self.csv_path)
                # Process the data
                if 'date' in self.df.columns:
                    self.df['date'] = pd.to_datetime(self.df['date'])
                if 'amount' in self.df.columns:
                    self.df['amount'] = pd.to_numeric(self.df['amount'], errors='coerce')
                # Add currency conversion
                if 'currency' in self.df.columns:
                    self.df['original_amount'] = self.df['amount']
                    self.df['original_currency'] = self.df['currency']
                    for idx, row in self.df.iterrows():
                        if row['currency'] == 'USD':
                            self.df.at[idx, 'amount'] = row['amount'] * self.exchange_rate
                    self.df['currency'] = 'KZT'
            except Exception as e2:
                print(f"Error loading data after creating sample: {e2}")
                self.df = pd.DataFrame()
    
    def analyze_spending_patterns(self, query: str = "") -> str:
        """Analyze spending patterns and make intelligent assumptions"""
        if self.df.empty:
            return "No transaction data available for analysis."
        
        analysis = []
        
        # Basic statistics
        total_transactions = len(self.df)
        total_spending = self.df['amount'].sum()
        avg_transaction = self.df['amount'].mean()
        
        analysis.append(f"📊 **Transaction Overview:**")
        analysis.append(f"- Total transactions: {total_transactions}")
        analysis.append(f"- Total spending: ₸{total_spending:.2f}")
        analysis.append(f"- Average transaction: ₸{avg_transaction:.2f}")
        
        # Category analysis
        if 'category' in self.df.columns:
            category_spending = self.df.groupby('category')['amount'].agg(['sum', 'count', 'mean'])
            top_categories = category_spending.sort_values('sum', ascending=False).head(3)
            
            analysis.append(f"\n💰 **Top Spending Categories:**")
            for category, data in top_categories.iterrows():
                analysis.append(f"- {category.title()}: ₸{data['sum']:.2f} ({data['count']} transactions)")
        
        # Time-based patterns
        if 'date' in self.df.columns:
            self.df['day_of_week'] = self.df['date'].dt.day_name()
            self.df['hour'] = self.df['date'].dt.hour
            
            # Most active day
            daily_spending = self.df.groupby('day_of_week')['amount'].sum()
            most_active_day = daily_spending.idxmax()
            
            analysis.append(f"\n📅 **Spending Patterns:**")
            analysis.append(f"- Most active spending day: {most_active_day}")
            analysis.append(f"- Daily average: ₸{daily_spending.mean():.2f}")
        
        # Smart assumptions
        assumptions = self._make_smart_assumptions()
        if assumptions:
            analysis.append(f"\n🤖 **AI Insights & Assumptions:**")
            analysis.extend(assumptions)
        
        return "\n".join(analysis)
    
    def _make_smart_assumptions(self) -> List[str]:
        """Generate intelligent assumptions based on transaction patterns"""
        assumptions = []
        
        if self.df.empty:
            return assumptions
        
        # Assumption 1: Budget categories
        if 'category' in self.df.columns:
            category_spending = self.df.groupby('category')['amount'].sum()
            total_spending = category_spending.sum()
            
            for category, amount in category_spending.items():
                percentage = (amount / total_spending) * 100
                if percentage > 30:
                    assumptions.append(f"🔍 High focus on {category} ({percentage:.1f}% of budget) - consider if this aligns with your priorities")
                elif percentage < 5:
                    assumptions.append(f"💡 Low spending on {category} ({percentage:.1f}%) - potential area for lifestyle improvement")
        
        # Assumption 2: Frequency patterns
        if 'description' in self.df.columns:
            recurring_merchants = self.df['description'].value_counts()
            frequent_merchants = recurring_merchants[recurring_merchants > 2]
            
            if not frequent_merchants.empty:
                assumptions.append(f"🔄 You frequently shop at: {', '.join(frequent_merchants.head(3).index.tolist())}")
        
        # Assumption 3: Spending behavior
        high_value_threshold = self.df['amount'].quantile(0.8)
        high_value_transactions = self.df[self.df['amount'] > high_value_threshold]
        
        if len(high_value_transactions) > 0:
            avg_high_value = high_value_transactions['amount'].mean()
            assumptions.append(f"💳 Your high-value purchases average ₸{avg_high_value:.2f} - these might be planned major expenses")
        
        # Assumption 4: Cash flow patterns
        if 'date' in self.df.columns and len(self.df) > 7:
            recent_week = self.df[self.df['date'] >= (self.df['date'].max() - timedelta(days=7))]
            older_week = self.df[self.df['date'] <= (self.df['date'].max() - timedelta(days=14))]
            
            if not recent_week.empty and not older_week.empty:
                recent_avg = recent_week['amount'].mean()
                older_avg = older_week['amount'].mean()
                
                if recent_avg > older_avg * 1.2:
                    assumptions.append("📈 Recent spending is higher than usual - monitor for budget impact")
                elif recent_avg < older_avg * 0.8:
                    assumptions.append("📉 Recent spending is lower than usual - good budget control")
        
        return assumptions
    
    def predict_future_spending(self, days: int = 30) -> str:
        """Predict future spending based on historical patterns"""
        if self.df.empty:
            return "No data available for predictions."
        
        # Calculate daily average
        if 'date' in self.df.columns:
            date_range = (self.df['date'].max() - self.df['date'].min()).days
            if date_range > 0:
                daily_avg = self.df['amount'].sum() / date_range
                predicted_spending = daily_avg * days
                
                return f"📊 **{days}-Day Spending Prediction:**\n" \
                       f"Based on your historical average of ₸{daily_avg:.2f}/day,\n" \
                       f"you might spend approximately ₸{predicted_spending:.2f} over the next {days} days."
        
        # Fallback to simple average
        avg_transaction = self.df['amount'].mean()
        transactions_per_day = len(self.df) / 30  # Assume 30-day period
        predicted_spending = avg_transaction * transactions_per_day * days
        
        return f"📊 **{days}-Day Spending Prediction:**\n" \
               f"Estimated spending: ₸{predicted_spending:.2f}"

class EnhancedFinanceAgent:
    """
    Enhanced Finance Agent with CSV analysis and intelligent assumptions
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.3,
        csv_path: str = TX_PATH
    ):
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model=model,
            temperature=temperature
        )
        
        # Save CSV path
        self.csv_path = csv_path
        
        # Initialize analyzer
        self.analyzer = TransactionAnalyzer(csv_path)
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Create tools
        self.tools = self._create_tools()
        
        # Create the agent
        self.agent = self._create_agent()

    def _create_tools(self) -> List[Tool]:
        """Create tools for the agent"""
        
        # 1. Direct CSV Agent Tool - for direct CSV operations
        try:
            from langchain_experimental.agents.agent_toolkits import create_csv_agent
        except ImportError:
            from langchain.agents.agent_toolkits import create_csv_agent
            
        csv_agent = create_csv_agent(
            llm=self.llm,
            path=self.csv_path,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            allow_dangerous_code=True
        )
        
        def csv_query(query: str) -> str:
            """Query the CSV data directly"""
            try:
                return csv_agent.run(query)
            except Exception as e:
                return f"Error querying CSV: {str(e)}"
        
        # 2. Datetime Tool - for current date and time information
        def get_current_datetime(format_str: str = "") -> str:
            """
            Get the current date and time.
            
            Args:
                format_str: Optional format string (e.g., '%Y-%m-%d' for date only)
                
            Returns:
                Current date and time information
            """
            now = datetime.now()
            
            if format_str:
                try:
                    return now.strftime(format_str)
                except Exception as e:
                    return f"Error formatting date: {str(e)}. Current datetime: {now.isoformat()}"
            
            # Default comprehensive information
            current_info = {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "day_of_week": now.strftime("%A"),
                "month": now.strftime("%B"),
                "year": now.year,
                "hour": now.hour,
                "minute": now.minute,
                "second": now.second,
                "timestamp": now.timestamp(),
                "iso_format": now.isoformat()
            }
            
            # Format as readable string
            result = []
            result.append(f"📅 Current Date: {current_info['date']} ({current_info['day_of_week']})")
            result.append(f"🕒 Current Time: {current_info['time']}")
            result.append(f"📆 Month/Year: {current_info['month']} {current_info['year']}")
            
            return "\n".join(result)
        
        # 3. Pattern Analysis Tool - for spending pattern analysis
        def pattern_analysis(query: str) -> str:
            """Analyze spending patterns and make assumptions"""
            return self.analyzer.analyze_spending_patterns(query)
        
        # 4. Prediction Tool - for future spending predictions
        def spending_prediction(query: str) -> str:
            """Predict future spending patterns"""
            # Extract number of days from query if present
            import re
            days_match = re.search(r'(\d+)\s*days?', query.lower())
            days = int(days_match.group(1)) if days_match else 30
            return self.analyzer.predict_future_spending(days)
        
        # 5. Create a dedicated CSV Agent Tool
        csv_file_agent = create_csv_agent(
            llm=self.llm,
            path=self.csv_path,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            allow_dangerous_code=True
        )
        
        def csv_file_query(query: str) -> str:
            """Query the CSV file directly using a dedicated agent"""
            try:
                return csv_file_agent.run(query)
            except Exception as e:
                return f"Error querying CSV file: {str(e)}"
        
        return [
            Tool(
                name="CSV_Query",
                func=csv_query,
                description="Query transaction data directly from CSV. Use for specific data retrieval, calculations, and filtering."
            ),
            Tool(
                name="Current_DateTime",
                func=get_current_datetime,
                description="Get the current date and time information. Can format with optional format string parameter."
            ),
            Tool(
                name="Pattern_Analysis",
                func=pattern_analysis,
                description="Analyze spending patterns and generate intelligent insights and assumptions about financial behavior."
            ),
            Tool(
                name="Spending_Prediction",
                func=spending_prediction,
                description="Predict future spending based on historical patterns. Mention number of days for custom predictions."
            ),
            Tool(
                name="CSV_File_Agent",
                func=csv_file_query,
                description="Advanced CSV file analysis tool. Query the CSV file with natural language to extract insights, run calculations, or answer specific questions about the data."
            )
        ]

    def _create_agent(self):
        """Create the enhanced agent with multiple tools"""
        system_message = f"""You are FinSight AI, an intelligent financial advisor chatbot. You have access to transaction data and several tools:

1. CSV_Query: Query transaction data directly using structured queries
2. Current_DateTime: Get current date and time information to provide time-aware insights
3. Pattern_Analysis: Analyze spending patterns and make intelligent assumptions
4. Spending_Prediction: Predict future spending based on historical patterns
5. CSV_File_Agent: Advanced CSV file analysis tool. Query the CSV file with natural language to extract insights, run calculations, or answer specific questions about the data.

Currency Conversion Information:
- The system automatically converts all transactions to Kazakhstan Tenge (₸, KZT)
- The current exchange rate used is {self.analyzer.exchange_rate:.2f} KZT per 1 USD
- All financial amounts are displayed in Tenge (₸) after conversion
- Original currency values are preserved in the data for reference

Important capabilities:
- You can call tools in sequence if needed - for example, get the current date first, then use it to analyze patterns
- Always consider the current date when making time-sensitive recommendations
- Combine insights from multiple tools to provide comprehensive advice

Always provide helpful, actionable insights. When users ask about their finances:
- Use specific data from their transactions
- Make intelligent assumptions about their spending habits
- Consider seasonal patterns or date-specific events
- Provide practical recommendations
- Be conversational and friendly

If a user asks a general question, try to relate it back to their actual transaction data when possible."""

        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=self.memory,
            verbose=True,
            agent_kwargs={
                "system_message": SystemMessage(content=system_message)
            }
        )

    def process_message(self, message: str) -> str:
        """Process user message and return response"""
        try:
            response = self.agent.run(message)
            return response
        except Exception as e:
            return f"I encountered an error processing your message: {str(e)}. Please try rephrasing your question."

# Factory function
def create_finance_agent(
    api_key: Optional[str] = None,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
    csv_path: str = TX_PATH
) -> EnhancedFinanceAgent:
    """Create an enhanced finance agent with multiple tools"""
    return EnhancedFinanceAgent(
        api_key=api_key, 
        model=model, 
        temperature=temperature,
        csv_path=csv_path
    )