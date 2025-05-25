import streamlit as st
from agents.langchain_agents import create_finance_agent
import os

def render_chat_interface():
    """Render AI-powered chat interface"""
    st.markdown("### 💬 Chat with FinSight AI")
    
    # Initialize session state for chat
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Check for OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        api_key_input = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key_input:
            st.session_state.OPENAI_API_KEY = api_key_input
            openai_api_key = api_key_input
            st.success("API key saved!")
            st.rerun()
        else:
            st.warning("Please enter your OpenAI API key to continue.")
            return
    

    
    # Initialize chat agent if not already done
    agent_key = f"finance_agent_gpt-4.1-mini"
    if "chat_agent" not in st.session_state or st.session_state.get("current_agent") != agent_key:
        try:
            # Initialize the finance agent with the new API
            st.session_state.chat_agent = create_finance_agent(
                api_key=openai_api_key,
                model="gpt-4.1-mini"
            )
            
            # Store current agent type
            st.session_state.current_agent = agent_key
            
            # Clear chat history when switching agents
            st.session_state.chat_messages = []
            
        except Exception as e:
            st.error(f"Error initializing AI chat: {str(e)}")
            return
    
    # Apply custom CSS for better chat styling
    st.markdown("""
    <style>
    .stChatMessage {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        max-width: 85%;
    }
    .stChatMessage[data-role="assistant"] {
        background-color: #f0f2f6;
        margin-right: auto;
    }
    .stChatMessage[data-role="user"] {
        background-color: #e6f7ff;
        margin-left: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Show agent capabilities
    with st.expander("What can this finance agent do?"):
        st.markdown("""
        **FinSight AI** can:
        - Answer questions about your transaction data
        - Perform calculations on your financial information
        - Update category configurations
        - Analyze spending patterns and trends
        - Provide insights about your finances
        
        Example questions:
        - "What were my top 5 expenses last month?"
        - "How much did I spend on groceries?"
        - "Update the 'food' category to 'Dining'"
        - "Calculate my average daily spending in March"
        - "Show me transactions over $100"
        """)
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask a question about your finances...")
    
    # Process user input
    if user_input:
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chat_agent.process_message(user_input)
                    st.write(response)
                    
                    # Add AI response to chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error processing your message: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})