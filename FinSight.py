import streamlit as st
from datetime import date
import pandas as pd

from core.data_manager import (
    initialize_session_state, 
    save_transaction, 
    load_transactions
)
from app.components.ui_components import (
    render_transaction_type_selector,
    render_transaction_form,
    render_recent_transactions,
    render_sidebar_summary
)
from agents.chat_assistant import render_chat_interface


def reset_form_state():
    """Reset form-related session state variables"""
    st.session_state.selected_type = None
    st.session_state.selected_currency = None
    st.session_state.selected_category = None
    st.session_state.form_submitted = False


def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="FinSight - AI-Powered Finance Tracker",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("💰 FinSight AI")
    st.markdown("### Your AI-Powered Financial Assistant")
    
    # Transaction type selection
    render_transaction_type_selector()
    
    st.markdown("---")
    
    # Render appropriate form based on selection
    if st.session_state.selected_type in ["expense", "income"]:
        # Handle form submission
        transaction_data = render_transaction_form(st.session_state.selected_type)
        
        if transaction_data:
            success = save_transaction(**transaction_data)
            if success:
                st.success(f"✅ {transaction_data['transaction_type'].title()} saved successfully to data/transactions.csv!")
                # Trigger AI analysis for new transaction
                if transaction_data['transaction_type'] == 'expense':
                    goal_warning = st.session_state.goal_agent.check_spending_warning(
                        transaction_data['category'], 
                        transaction_data['amount']
                    )
                    if goal_warning:
                        st.warning(f"⚠️ {goal_warning}")
                
                reset_form_state()
                st.rerun()
            else:
                st.error("❌ Failed to save transaction!")
        
        # Close form button
        if st.button("❌ Close Form", type="secondary"):
            reset_form_state()
            st.rerun()
    else:
        st.info("👆 Select a transaction type above or import a bank statement to get started!")
    
    # Recent transactions
    st.markdown("---")
    render_recent_transactions()
    
    # Chat interface after recent transactions
    st.markdown("---")
    render_chat_interface()
    
    # Sidebar summary
    render_sidebar_summary()


if __name__ == "__main__":
    main() 