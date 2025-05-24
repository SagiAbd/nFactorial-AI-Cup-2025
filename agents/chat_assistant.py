import streamlit as st
from core.data_manager import load_transactions, get_financial_summary, get_category_spending
from agents.langchain_agents import ChatAssistantAgent
import time


def render_chat_interface():
    """Render AI-powered chat interface"""
    st.markdown("### 💬 Chat with FinSight AI")
    
    # Initialize chat assistant agent
    if "chat_agent" not in st.session_state:
        try:
            st.session_state.chat_agent = ChatAssistantAgent()
        except Exception as e:
            st.error(f"Failed to initialize chat agent: {str(e)}")
            st.info("The chat interface requires proper API configuration. Please check your OpenAI API key.")
    
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
        
        # Show the user message immediately
        with st.chat_message("user"):
            st.write(user_input)
        
        # Generate response (with memory support)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Check if we have an agent
                    if "chat_agent" in st.session_state:
                        # Get financial context if needed
                        context = {}
                        try:
                            transactions = load_transactions()
                            if not transactions.empty:
                                # Add minimal context without overloading
                                context = {
                                    "total_transactions": len(transactions),
                                    "has_data": True
                                }
                        except:
                            pass
                            
                        response = st.session_state.chat_agent.generate_response(user_input, context)
                    else:
                        # Fallback to simple response
                        response = "I'm a simple financial assistant. Currently in development mode, but I'd be happy to help when fully implemented!"
                except Exception as e:
                    response = f"Sorry, I encountered an error: {str(e)}"
                
                # Add assistant response to chat and display it
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
    
    # Add a clear chat button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🗑️ Clear Chat"):
            # Keep only the welcome message
            st.session_state.messages = [st.session_state.messages[0]]
            st.rerun()
    
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