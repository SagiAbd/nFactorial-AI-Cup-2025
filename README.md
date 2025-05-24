# FinSight AI - Financial Assistant

An AI-powered personal finance tracker with LangChain integration for insights and financial guidance.

## Project Structure

```
/
├── app/                  # Main application code
│   ├── __init__.py
│   ├── main.py           # Main application logic
│   ├── pages/            # Additional Streamlit pages
│   └── components/       # UI components
│       ├── __init__.py
│       └── ui_components.py
├── core/                 # Core business logic
│   ├── __init__.py
│   └── data_manager.py   # Data handling and storage
├── agents/               # AI agents
│   ├── __init__.py
│   ├── langchain_agents.py
│   ├── categorizer.py
│   ├── clarifier.py
│   └── recommendations.py
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── budget_math.py
│   ├── charts.py
│   └── chat_assistant.py
├── data/                 # Data storage
├── .streamlit/           # Streamlit configuration
├── requirements.txt      # Dependencies
└── streamlit_app.py      # Entry point
```

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install requirements:
   ```
   pip install -r requirements.txt
   ```

3. Set up API keys:
   - Create a `.env` file in the root directory with your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - For Streamlit Cloud deployment, add the key to the Streamlit secrets.

4. Run the application:
   ```
   streamlit run streamlit_app.py
   ```

## Features

- 💰 Track income and expenses
- 📊 Visualize spending patterns
- 🎯 Set and monitor financial goals
- 🤖 AI-powered insights and recommendations
- 💬 Chat interface for financial guidance

## Deployment

To deploy to Streamlit Cloud:

1. Push your code to a GitHub repository
2. Connect your repository to Streamlit Cloud
3. Add your OPENAI_API_KEY to the Streamlit secrets
4. Deploy the app

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for AI features

When running on Streamlit Cloud, the app will automatically use the key from Streamlit secrets. 