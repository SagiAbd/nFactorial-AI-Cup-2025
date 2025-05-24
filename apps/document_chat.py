"""
Document Chat Application
Chat interface for processing financial documents.
"""
import os
import streamlit as st
from datetime import datetime
import pandas as pd

# Import agent
from agents.document_agent import DocumentAgent

# Set page configuration
st.set_page_config(
    page_title="Financial Document Chat",
    page_icon="💰",
    layout="wide"
)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = DocumentAgent()
    
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
    
if "has_document" not in st.session_state:
    st.session_state.has_document = False
    
if "transactions" not in st.session_state:
    st.session_state.transactions = []

# Header
st.title("Financial Document Assistant 💼")
st.write("Upload financial documents and chat with the AI about your transactions.")

# Sidebar with document upload
with st.sidebar:
    st.header("Document Upload")
    
    uploaded_file = st.file_uploader(
        "Upload a financial document",
        type=["pdf", "jpg", "jpeg", "png"],
        help="Upload a bank statement or receipt"
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary"):
            with st.spinner("Processing document..."):
                # Process the document
                result = st.session_state.agent.process_document(uploaded_file)
                
                if result.get("success", False):
                    st.session_state.has_document = True
                    st.session_state.transactions = result.get("transactions", [])
                    
                    # Add system message to chat
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": f"✅ Document processed successfully: {uploaded_file.name}"
                    })
                else:
                    error_msg = result.get("error", "Unknown error")
                    st.error(f"Error processing document: {error_msg}")
    
    # Transaction summary if available
    if st.session_state.has_document:
        summary = st.session_state.agent.get_transaction_summary()
        
        if summary.get("success", False):
            st.subheader("Document Summary")
            st.write(f"Total Transactions: {summary.get('total', 0)}")
            st.write(f"Income Transactions: {summary.get('income_count', 0)}")
            st.write(f"Expense Transactions: {summary.get('expense_count', 0)}")
            
            if st.button("Save All Transactions"):
                with st.spinner("Saving transactions..."):
                    save_result = st.session_state.agent.save_transactions()
                    
                    if save_result.get("success", False):
                        st.success(f"✅ Saved {save_result.get('saved_count', 0)} transactions")
                    else:
                        st.error(f"Error saving transactions: {save_result.get('error', 'Unknown error')}")

# Main area with tabs
tab1, tab2 = st.tabs(["Chat", "Transactions"])

with tab1:
    # Chat interface
    st.subheader("Chat with the AI")
    
    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your transactions..."):
        # Add user message to chat
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.get_chat_completions(prompt)
                
                if response.get("success", False):
                    message = response.get("message", "I don't know how to respond to that.")
                    st.write(message)
                    
                    # Add to chat history
                    st.session_state.chat_messages.append({"role": "assistant", "content": message})
                else:
                    error_msg = response.get("error", "Unknown error")
                    st.error(f"Error getting response: {error_msg}")

with tab2:
    # Transactions view
    st.subheader("Transactions")
    
    if st.session_state.has_document and st.session_state.transactions:
        # Add categorize button
        if st.button("Auto-Categorize"):
            with st.spinner("Categorizing transactions..."):
                result = st.session_state.agent.categorize_transactions()
                
                if result.get("success", False):
                    st.success(f"✅ Categorized {result.get('updated_count', 0)} transactions")
                    
                    # Refresh transactions list
                    summary = st.session_state.agent.get_transaction_summary()
                    if summary.get("success", False):
                        st.session_state.transactions = summary.get("transactions", [])
                else:
                    st.error(f"Error categorizing transactions: {result.get('error', 'Unknown error')}")
        
        # Create DataFrame from transactions
        df = pd.DataFrame(st.session_state.transactions)
        
        # Display transactions
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No transactions yet. Upload and process a document to see transactions.")

# Footer
st.markdown("---")
st.caption("Financial Document Assistant | Powered by AI") 