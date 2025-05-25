import os
import json
import pandas as pd
from typing import Optional, List
from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.tools import Tool

# Paths
TX_PATH = "data/transactions.csv"
CONFIG_PATH = "config.json"

# -----------------------------
# Utility functions
# -----------------------------

def load_config() -> dict:
    """Load the configuration file or return default structure."""
    if not os.path.exists(CONFIG_PATH):
        return {"categories": {}}
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def save_config(config: dict) -> None:
    """Save the configuration dict to disk."""
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def update_config_category(key: str, value: Optional[str] = None) -> str:
    """
    Tool function to get or set a category in the config.
    If `value` is None, returns current category value.
    Otherwise, updates the category and saves.
    """
    if not key:
        return "Error: A category key is required."

    config = load_config()
    categories = config.setdefault("categories", {})

    if value is None:
        return f"Current value for '{key}': '{categories.get(key, 'Not set')}'"

    categories[key] = value
    save_config(config)
    return f"Updated category '{key}' to '{value}'"

# Wrap config tool
config_tool = Tool(
    name="update_config",
    func=update_config_category,
    description="Get or set a category in the config.json file. Usage: update_config(key, [value])"
)

# -----------------------------
# Agent builder
# -----------------------------
class ChatAssistantAgent:
    """
    A unified Chat Assistant Agent that can:
      - Answer queries against a transactions CSV
      - Update/check categories in a config file
      - Perform LLM-based math operations
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.0
    ):
        # Initialize LLM
        llm_class = ChatOpenAI if "gpt-4" in model or "gpt-3.5" in model else OpenAI
        self.llm = llm_class(
            openai_api_key=api_key,
            model=model,
            temperature=temperature
        )

        # Ensure sample data
        self._ensure_sample_transactions()

        # Build tools
        tools = self._gather_tools()

        # Initialize agent with memory and ReAct template
        from langchain.memory import ConversationBufferMemory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.agent = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            memory=memory,
            verbose=True
        )

    def _ensure_sample_transactions(self) -> None:
        """
        Creates a sample CSV file if none exists at TX_PATH.
        """
        if os.path.exists(TX_PATH):
            return

        os.makedirs(os.path.dirname(TX_PATH), exist_ok=True)
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5).strftime('%Y-%m-%d'),
            'description': ['Grocery', 'Dining', 'Fuel', 'Online', 'Grocery'],
            'amount': [85.75, 45.20, 65.30, 120.50, 55.60],
            'type': ['expense'] * 5,
            'category': ['groceries', 'dining', 'transport', 'shopping', 'groceries'],
            'currency': ['USD'] * 5
        })
        df.to_csv(TX_PATH, index=False)

    def _gather_tools(self) -> List[Tool]:
        """
        Load the CSV-reading tool, wrap it with a clear name/description,
        add the config tool, and any extra tools (e.g., math).
        """
        tools: List[Tool] = []

        # 1) CSV tool via create_csv_agent
        try:
            from langchain_experimental.agents.agent_toolkits import create_csv_agent
        except ImportError:
            from langchain.agents.agent_toolkits import create_csv_agent

        csv_agent = create_csv_agent(
            llm=self.llm,
            path=TX_PATH,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            allow_dangerous_code=True
        )
        # Wrap raw tool for naming clarity
        raw_csv_tool = csv_agent.tools[0]
        csv_tool = Tool(
            name="transactions_csv",
            func=raw_csv_tool.run,
            description=(
                "Use this to query the transactions.csv file. "
                "Filter by month/date and aggregate `amount` (sum, average, etc.)."
            )
        )
        tools.append(csv_tool)

        # 2) Config tool
        tools.append(config_tool)

        # 3) LLM math tools
        try:
            math_tools = load_tools(["llm-math"], llm=self.llm)
            tools.extend(math_tools)
        except Exception:
            pass  # skip if unavailable

        return tools

    def process_message(self, message: str) -> str:
        """Send a message to the agent and get its response."""
        return self.agent.run(message)

# Factory function

def create_finance_agent(
    api_key: Optional[str] = None,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.0
) -> ChatAssistantAgent:
    """Helper to instantiate the ChatAssistantAgent."""
    return ChatAssistantAgent(api_key=api_key, model=model, temperature=temperature)
