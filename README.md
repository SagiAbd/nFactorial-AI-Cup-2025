# FinSight AI - Financial Assistant

An AI-powered personal finance tracker with LangChain integration for insights and financial guidance.

## Project Structure

```
/
├── app/                  # Main application code
│   ├── __init__.py
│   ├── main.py           # Main application entry point
│   └── components/       # UI components
│       ├── __init__.py
│       └── ui_components.py
├── pages/                # Streamlit pages (multi-page app)
│   ├── 1_Financial_Dashboard.py
│   ├── 2_Transactions.py
│   ├── 3_Budget_Planning.py
│   ├── 4_Analytics.py
│   └── 5_New_Transaction.py
├── core/                 # Core business logic
│   ├── __init__.py
│   └── data_manager.py   # Data handling and storage
├── agents/               # AI agents
│   ├── __init__.py
│   ├── langchain_agents.py
│   ├── bank_statement_processor.py  # Bank statement upload processor
│   ├── categorizer.py
│   ├── clarifier.py
│   └── recommendations.py
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── budget_math.py
│   ├── charts.py
│   └── chat_assistant.py
├── data/                 # Data storage
│   └── uploads/          # Storage for uploaded file records
├── .streamlit/           # Streamlit configuration
├── requirements.txt      # Dependencies
└── FinSight.py           # Main application entry point
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
   streamlit run FinSight.py
   ```

## Features

- 💰 Track income and expenses
- 📊 Visualize spending patterns
- 🎯 Set and monitor financial goals
- 🤖 AI-powered insights and recommendations
- 💬 Chat interface for financial guidance
- 📂 Bank statement and receipt upload
  - Upload bank statements and receipts directly
  - Supports CSV, PDF, JPG, PNG, XLS and XLSX formats
  - View confirmation of uploaded files

## Bank Statement Upload

The application supports direct upload of financial documents:

1. On the main page, select the "Import" option
2. Use the file uploader to select your bank statement or receipt files
3. Click "Process Documents" to upload the files
4. A success message will confirm your documents have been uploaded
5. The uploaded files are recorded in `data/uploads/uploads_record.csv`

## Deployment

To deploy to Streamlit Cloud:

1. Push your code to a GitHub repository
2. Connect your repository to Streamlit Cloud
3. Add your OPENAI_API_KEY to the Streamlit secrets
4. Deploy the app

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for AI features

When running on Streamlit Cloud, the app will automatically use the key from Streamlit secrets.

# Memory System

FinSight includes an enhanced memory system that allows the AI assistant to remember past conversations with users. This enables more natural and context-aware interactions.

## Memory Features

- **Persistent Memory**: Conversations are stored between sessions
- **Conversation Summarization**: Long conversations are automatically summarized
- **User-Specific Memory**: Each user gets their own memory space
- **Memory Retrieval**: The agent can access and reference past conversations
- **Memory Clearing**: Users can clear their conversation history if desired

## How to Enable Enhanced Memory

The enhanced memory system is enabled by default through the configuration in `.streamlit/config.toml`. You can toggle this setting by changing:

```toml
[feature]
USE_ENHANCED_AGENT = true
```

## Testing Memory

You can test the memory system by running:

```bash
python test_memory.py
```

This will verify that the memory persistence and retrieval features are working correctly.

## Memory Implementation

The memory system uses LangChain's memory components:

- `FileChatMessageHistory`: Persists conversations to disk
- `ConversationBufferMemory`: Stores recent conversation turns in full
- `ConversationSummaryMemory`: Creates summaries of longer conversations to maintain context

All memory data is stored in the `data/memory` directory, with each user having their own memory file.

## Memory Limitations

- The memory system requires OpenAI API access to function
- Memory files can grow large over time, so periodic maintenance may be needed
- Memory summarization consumes additional API tokens 

# Memory System Updates

The memory system has been updated to use the latest LangChain modules:

- Updated imports to use `langchain_community.chat_message_histories` instead of deprecated paths
- Fixed conversation chain configuration for better compatibility
- Added fallback mechanisms in case of initialization errors
- Memory system now gracefully handles API key issues

The memory system also includes:

- A test script (`test_memory.py`) to validate memory persistence
- A simple agent initialization test (`test_agent_init.py`)
- Better error handling and fallback to simple mode when needed

## Debugging Memory Issues

If you encounter issues with the memory system:

1. Make sure your OpenAI API key is set properly
2. Run `python test_agent_init.py` to verify basic agent initialization
3. Check the error messages for specific issues
4. Memory files are stored in `data/memory/` and can be inspected if needed 