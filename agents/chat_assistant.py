import streamlit as st
from agents.langchain_agents import create_finance_agent
import os

def render_chat_interface():
    """Render enhanced AI-powered chat interface"""
    st.markdown("### 💬 Chat with FinSight AI Pro")
    st.markdown("*Enhanced with intelligent pattern analysis and predictions*")
    
    # Initialize session state for chat
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Check for OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY") or st.session_state.get("OPENAI_API_KEY")
    
    if not openai_api_key:
        st.info("🔑 Enter your OpenAI API key to start chatting with your intelligent financial advisor")
        api_key_input = st.text_input("Enter your OpenAI API key:", type="password")
        if api_key_input:
            st.session_state.OPENAI_API_KEY = api_key_input
            openai_api_key = api_key_input
            st.success("✅ API key saved! You can now start chatting.")
            st.rerun()
        else:
            st.stop()
    
    # Initialize chat agent if not already done
    agent_key = f"enhanced_finance_agent_gpt-4.1-mini"
    if "chat_agent" not in st.session_state or st.session_state.get("current_agent") != agent_key:
        try:
            with st.spinner("🤖 Initializing your AI financial advisor..."):
                # Initialize the enhanced finance agent
                st.session_state.chat_agent = create_finance_agent(
                    api_key=openai_api_key,
                    model="gpt-4.1-mini"
                )
                
                # Store current agent type
                st.session_state.current_agent = agent_key
                
                # Clear chat history when switching agents
                st.session_state.chat_messages = []
                
                st.success("🎉 FinSight AI Pro is ready!")
            
        except Exception as e:
            st.error(f"❌ Error initializing AI chat: {str(e)}")
            st.stop()
    
    # Apply custom CSS for better chat styling
    st.markdown("""
    <style>
    .stChatMessage {
        padding: 0.75rem 1rem;
        border-radius: 0.75rem;
        margin-bottom: 0.75rem;
        max-width: 85%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stChatMessage[data-role="assistant"] {
        background: linear-gradient(135deg, #f0f2f6 0%, #e8eaf0 100%);
        margin-right: auto;
        border-left: 4px solid #0066cc;
    }
    .stChatMessage[data-role="user"] {
        background: linear-gradient(135deg, #e6f7ff 0%, #d1f2ff 100%);
        margin-left: auto;
        border-right: 4px solid #1890ff;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create two columns for capabilities and quick actions
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Show enhanced agent capabilities
        with st.expander("🚀 What can FinSight AI Pro do?", expanded=False):
            st.markdown("""
            **FinSight AI Pro** is your intelligent financial advisor with three powerful tools:
            
            **📊 Direct Data Analysis:**
            - Query specific transactions and amounts
            - Calculate spending by category, date, or merchant
            - Find patterns in your financial data
            
            **🤖 Intelligent Pattern Recognition:**
            - Analyze your spending habits automatically
            - Generate insights about your financial behavior
            - Make smart assumptions about your lifestyle
            
            **🔮 Predictive Analytics:**
            - Forecast future spending based on your patterns
            - Predict budget needs for upcoming periods
            - Estimate costs for different scenarios
            
            **Example questions:**
            - "What insights can you give me about my spending?"
            - "Predict my spending for the next 2 weeks"
            - "What assumptions can you make about my lifestyle?"
            - "How much do I typically spend on weekends?"
            """)
    
    with col2:
        # Quick action suggestions
        st.markdown("#### 💡 Quick Actions")
        
        if st.button("📈 Analyze My Patterns", use_container_width=True):
            quick_message = "Analyze my spending patterns and give me insights about my financial habits"
            st.session_state.chat_messages.append({"role": "user", "content": quick_message})
            
        if st.button("🔮 Predict Future Spending", use_container_width=True):
            quick_message = "Predict my spending for the next 30 days based on my transaction history"
            st.session_state.chat_messages.append({"role": "user", "content": quick_message})
            
        if st.button("💰 Budget Breakdown", use_container_width=True):
            quick_message = "Show me a breakdown of my spending by category and give me budget recommendations"
            st.session_state.chat_messages.append({"role": "user", "content": quick_message})
    
    st.divider()
    
    # Display chat messages in a container
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your finances... 💬")
    
    # Process user input or quick actions
    if user_input or (st.session_state.chat_messages and 
                     st.session_state.chat_messages[-1]["role"] == "user" and 
                     len(st.session_state.chat_messages) > len([m for m in st.session_state.chat_messages if m["role"] == "assistant"])):
        
        # Get the latest user message
        if user_input:
            current_message = user_input
            # Add user message to chat history
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
        else:
            # This is from a quick action button
            current_message = st.session_state.chat_messages[-1]["content"]
            with st.chat_message("user"):
                st.markdown(current_message)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🧠 Analyzing your data and generating insights..."):
                try:
                    response = st.session_state.chat_agent.process_message(current_message)
                    
                    # Display response with better formatting
                    st.markdown(response)
                    
                    # Add AI response to chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"❌ I encountered an error: {str(e)}\n\nPlease try rephrasing your question or contact support if the issue persists."
                    st.error("Something went wrong while processing your request.")
                    st.markdown(error_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
    
    # Add footer with tips
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    💡 <strong>Pro Tips:</strong> Ask specific questions like "What are my top expenses?" or "Analyze my weekend spending patterns" for better insights!
    </div>
    """, unsafe_allow_html=True)

# Optional: Add sidebar with agent status and controls
def render_sidebar_controls():
    """Render sidebar with additional controls and information"""
    with st.sidebar:
        st.markdown("### 🤖 FinSight AI Pro Status")
        
        if "chat_agent" in st.session_state:
            st.success("✅ Agent Active")
            st.markdown("**Tools Available:**")
            st.markdown("- 📊 CSV Data Query")
            st.markdown("- 🤖 Pattern Analysis")
            st.markdown("- 🔮 Spending Prediction")
        else:
            st.warning("⏳ Agent Not Initialized")
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_messages = []
            st.rerun()
        
        # Export chat button
        if st.session_state.get("chat_messages"):
            chat_export = "\n".join([
                f"**{msg['role'].title()}:** {msg['content']}\n" 
                for msg in st.session_state.chat_messages
            ])
            st.download_button(
                label="📥 Export Chat",
                data=chat_export,
                file_name="finsight_chat_history.txt",
                mime="text/plain",
                use_container_width=True
            )

# Main function to run the enhanced interface
def main():
    st.set_page_config(
        page_title="FinSight AI Pro",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 FinSight AI Pro")
    st.markdown("*Your Intelligent Financial Advisor with Advanced Analytics*")
    
    # Render sidebar controls
    render_sidebar_controls()
    
    # Render main chat interface
    render_chat_interface()

if __name__ == "__main__":
    main()