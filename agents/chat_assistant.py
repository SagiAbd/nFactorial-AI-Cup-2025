import streamlit as st
from core.data_manager import load_transactions, get_financial_summary, get_category_spending
from agents.langchain_agents import ChatAssistantAgent
import time


def render_chat_interface():
    """Render AI-powered chat interface"""
    st.markdown("### 💬 Chat with FinSight AI")
    
    # Initialize chat assistant agent
    if "chat_agent" not in st.session_state:
        st.session_state.chat_agent = ChatAssistantAgent()
    
    # Initialize messages in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Hi! I'm your AI financial assistant. I can help you with:\n• 📊 Financial insights and analysis\n• 🎯 Goal tracking and recommendations\n• 💡 Spending patterns and trends\n• ❓ Any questions about your finances"
        })
    
    # Display chat messages directly in the UI
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Add a small separator
    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
    
    # Chat input 
    user_input = st.chat_input("Ask me about your finances, goals, or get insights...")
    
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Generate AI response using LangChain agent
        with st.spinner("Thinking..."):
            response = st.session_state.chat_agent.generate_response(user_input)
            
            # Add assistant response to chat
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()  # Rerun to update the chat display
    
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