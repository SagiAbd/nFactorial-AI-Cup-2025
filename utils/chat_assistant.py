import streamlit as st
from streamlit_extras.bottom_container import bottom
from core.data_manager import load_transactions, get_financial_summary, get_category_spending
from agents.langchain_agents import ChatAssistantAgent


def render_chat_interface():
    """Render AI-powered chat interface at the bottom"""
    with bottom():
        st.markdown("---")
        st.subheader("🤖 FinSight AI Assistant")
        
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
        
        # Display chat messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Chat input
        user_input = st.chat_input("Ask me about your finances, goals, or get insights...")
        
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Display user message immediately
            with st.chat_message("user"):
                st.write(user_input)
            
            # Generate AI response using LangChain agent
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.chat_agent.generate_response(user_input)
                
                # Add assistant response to chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Display assistant response
                with st.chat_message("assistant"):
                    st.write(response) 