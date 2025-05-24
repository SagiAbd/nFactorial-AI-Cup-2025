import streamlit as st
import pandas as pd
from datetime import date, time, datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
from streamlit_extras.mandatory_date_range import date_range_picker

# Add parent directory to path to enable imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.data_manager import (
    load_transactions, save_transaction, load_config, 
    get_financial_summary, get_category_spending, add_category
)
from utils.budget_math import calculate_budget_progress
from utils.charts import create_spending_trend_chart, create_category_distribution
from agents.bank_statement_processor import BankStatementProcessor


def render_transaction_type_selector():
    """Render the transaction type selector with buttons"""
    st.write("**Choose Transaction Type or Import**")
    
    # Initialize import mode in session state if not present
    if "import_mode" not in st.session_state:
        st.session_state.import_mode = False
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        expense_selected = st.button(
            "💸 Add Expense", 
            use_container_width=True, 
            type="secondary",
            disabled=st.session_state.selected_type == "expense"
        )
        if expense_selected:
            st.session_state.selected_type = "expense"
            st.session_state.selected_currency = None
            st.session_state.selected_category = None
            st.session_state.import_mode = False
            st.rerun()
    
    with col2:
        income_selected = st.button(
            "💰 Add Income", 
            use_container_width=True, 
            type="secondary",
            disabled=st.session_state.selected_type == "income"
        )
        if income_selected:
            st.session_state.selected_type = "income"
            st.session_state.selected_currency = None
            st.session_state.selected_category = None
            st.session_state.import_mode = False
            st.rerun()
    
    with col3:
        upload_button = st.button(
            "📁 Import",
            use_container_width=True,
            type="secondary",
            disabled=st.session_state.import_mode
        )
        if upload_button:
            st.session_state.import_mode = True
            st.session_state.selected_type = None
            st.rerun()
    
    # Display the file uploader when in import mode
    if st.session_state.import_mode:
        # Initialize processing options in session state if not present
        if "processing_options" not in st.session_state:
            st.session_state.processing_options = {
                "ask_clarification": True,
                "allow_new_categories": True,
                "date_range": "1 month"
            }
        
        uploaded_file = st.file_uploader(
            "Upload your bank statement or receipt images",
            type=["csv", "xlsx", "pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="financial_document_uploader"
        )
        
        # Add processing options with checkboxes
        st.write("**Processing Options**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.session_state.processing_options["ask_clarification"] = st.checkbox(
                "Ask for clarification on unclear transactions",
                value=True,
                help="The AI will ask questions about transactions it can't confidently categorize"
            )
            
            st.session_state.processing_options["allow_new_categories"] = st.checkbox(
                "Allow creating new categories",
                value=True,
                help="Automatically create new categories for transactions that don't fit existing ones"
            )
            
            # Add auto-save option
            st.session_state.processing_options["auto_save"] = st.checkbox(
                "Auto-save transactions",
                value=False,
                help="Automatically save transactions to data/transactions.csv without preview"
            )
        
        with col2:
            date_range_options = {
                "1 week": "Last week",
                "2 weeks": "Last 2 weeks",
                "1 month": "Last month",
                "3 months": "Last 3 months", 
                "6 months": "Last 6 months",
                "all": "All dates",
                "custom": "Custom range"
            }
            
            st.session_state.processing_options["date_range"] = st.selectbox(
                "Parse time range",
                options=list(date_range_options.keys()),
                format_func=lambda x: date_range_options[x],
                index=list(date_range_options.keys()).index("1 month"),
                help="Only process transactions within this time period"
            )
            
            # Show date range picker when "Custom range" is selected
            if st.session_state.processing_options["date_range"] == "custom":
                # Default start date is 1 month ago
                default_start = datetime.now() - timedelta(days=30)
                # Default end date is today
                default_end = datetime.now()
                
                # Use the date_range_picker component
                start_date, end_date = date_range_picker(
                    "Select custom date range",
                    default_start=default_start,
                    default_end=default_end,
                    max_date=datetime.now(),
                    min_date=datetime.now() - timedelta(days=365*2),  # Up to 2 years ago
                    format="YYYY-MM-DD"
                )
                
                # Store the selected dates in processing options
                st.session_state.processing_options["custom_start_date"] = start_date
                st.session_state.processing_options["custom_end_date"] = end_date
                
                # Show selected range as text
                days_selected = (end_date - start_date).days
                st.caption(f"Selected range: {days_selected} days ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
        
        # Add divider
        st.markdown("---")
        
        # Add a cancel button to exit import mode
        cancel_col, process_col = st.columns([1, 1])
        with cancel_col:
            if st.button("Cancel Import", type="secondary", use_container_width=True):
                st.session_state.import_mode = False
                st.rerun()
        
        if uploaded_file:
            with process_col:
                process_button = st.button("Process Documents", type="primary", use_container_width=True)
                
            if process_button:
                with st.spinner("Processing your documents..."):
                    # Initialize the bank statement processor
                    processor = BankStatementProcessor()
                    
                    # Pass processing options to the processor
                    processor.set_processing_options(st.session_state.processing_options)
                    
                    # Process the uploaded files
                    transactions_df = processor.process_files(uploaded_file)
                    
                    if transactions_df is not None and not transactions_df.empty:
                        # Show a preview of the processed transactions
                        st.success(f"📁 Successfully processed {len(uploaded_file)} document(s)")
                        
                        # Check if auto-save is enabled
                        if st.session_state.processing_options.get("auto_save", False):
                            with st.spinner("Auto-saving transactions..."):
                                success_count, error_count = processor.save_processed_transactions(transactions_df)
                                
                                if success_count > 0:
                                    st.success(f"✅ Automatically saved {success_count} transactions to data/transactions.csv!")
                                    if error_count > 0:
                                        st.warning(f"⚠️ Failed to save {error_count} transactions.")
                                    
                                    # Exit import mode
                                    st.session_state.import_mode = False
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to save any transactions.")
                        else:
                            # Display a preview of the processed transactions
                            with st.expander("Preview Processed Transactions", expanded=True):
                                st.dataframe(transactions_df)
                                
                                # Save button
                                if st.button("💾 Save All Transactions to data/transactions.csv", type="primary"):
                                    with st.spinner("Saving transactions..."):
                                        success_count, error_count = processor.save_processed_transactions(transactions_df)
                                        
                                        if success_count > 0:
                                            st.success(f"✅ Successfully saved {success_count} transactions to data/transactions.csv!")
                                            if error_count > 0:
                                                st.warning(f"⚠️ Failed to save {error_count} transactions.")
                                            
                                            # Exit import mode
                                            st.session_state.import_mode = False
                                            st.rerun()
                                        else:
                                            st.error("❌ Failed to save any transactions.")
                    else:
                        st.error("❌ Failed to process the uploaded documents. Please check the file format.")

def render_currency_selector():
    """Render currency selection using button selector style"""
    st.write("**Currency**")
    currencies = st.session_state.config["currencies"]
    
    # Set KZT as default currency if none is selected
    if st.session_state.selected_currency is None:
        st.session_state.selected_currency = "KZT"
    
    # Create a row of currency buttons
    cols = st.columns(len(currencies))
    for i, currency in enumerate(currencies):
        with cols[i]:
            is_selected = st.session_state.selected_currency == currency
            button_type = "primary" if is_selected else "secondary"
            
            if st.button(
                currency, 
                key=f"currency_{currency}_{st.session_state.selected_type}", 
                use_container_width=True,
                type=button_type
            ):
                st.session_state.selected_currency = currency
                st.rerun()

def render_category_selector(transaction_type):
    """Render category selection using button selector style"""
    st.write("**Category**")
    categories = st.session_state.config[f"{transaction_type}_categories"]
    
    # Calculate grid layout (max 4 per row)
    cols_per_row = min(4, len(categories))
    rows = [categories[i:i + cols_per_row] for i in range(0, len(categories), cols_per_row)]
    
    for row_cats in rows:
        cols = st.columns(len(row_cats))
        for i, cat in enumerate(row_cats):
            with cols[i]:
                is_selected = st.session_state.selected_category == cat['name']
                button_type = "primary" if is_selected else "secondary"
                
                if st.button(
                    f"{cat['icon']} {cat['name']}", 
                    key=f"cat_{transaction_type}_{cat['name']}", 
                    use_container_width=True,
                    type=button_type
                ):
                    st.session_state.selected_category = cat['name']
                    st.rerun()

def render_add_category_form(transaction_type):
    """Render form to add new category"""
    with st.expander("➕ Add New Category"):
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            new_cat_name = st.text_input("Category name", key=f"new_cat_name_{transaction_type}")
        with col2:
            new_cat_icon = st.text_input("Icon", placeholder="🏷️", key=f"new_cat_icon_{transaction_type}")
        with col3:
            st.write("")  # Empty space for alignment
            if st.button("Add", key=f"add_cat_{transaction_type}", use_container_width=True):
                if new_cat_name and new_cat_icon:
                    if add_category(transaction_type, new_cat_name, new_cat_icon):
                        st.success(f"✅ Added: {new_cat_icon} {new_cat_name}")
                        st.rerun()
                    else:
                        st.warning("Category already exists!")
                else:
                    st.warning("Please enter both name and icon!")




def render_transaction_form(transaction_type):
    """Render the transaction form for expense or income"""
    st.subheader(f"Add {transaction_type.title()}")
    
    # Amount input
    st.write("**Amount**")
    amount = st.number_input(
        "Amount", 
        min_value=0.01, 
        step=0.01, 
        format="%.2f", 
        label_visibility="collapsed",
        key=f"amount_{transaction_type}"
    )
    
    # Currency selection
    render_currency_selector()
    
    # Date input (without time)
    st.write("**Date**")
    transaction_date = st.date_input(
        "Date",
        value=date.today(),
        key=f"date_{transaction_type}",
        label_visibility="collapsed"
    )
    
    # Convert date to datetime for consistency with existing code
    transaction_date_time = datetime.combine(transaction_date, time(0, 0, 0))
    
    # Category selection
    render_category_selector(transaction_type)
    
    # Add new category option
    render_add_category_form(transaction_type)
    
    # Description input
    description = st.text_area(
        "**Description (Optional)**", 
        placeholder="Add any additional notes...",
        key=f"description_{transaction_type}",
        height=80
    )
    
    # Display selected values
    if st.session_state.selected_currency or st.session_state.selected_category:
        st.info(f"💱 **Currency:** {st.session_state.selected_currency or 'Not selected'} | "
               f"📂 **Category:** {st.session_state.selected_category or 'Not selected'}")
    
    # Save button (only show when all required fields are filled)
    if (amount > 0 and 
        st.session_state.selected_currency and 
        st.session_state.selected_category):
        
        st.write("")  # Space
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("💾 Save Transaction", use_container_width=True, type="primary"):
                return {
                    "transaction_date": transaction_date_time,
                    "amount": amount,
                    "category": st.session_state.selected_category,
                    "currency": st.session_state.selected_currency,
                    "transaction_type": transaction_type,
                    "description": description
                }
    
    return None


def render_recent_transactions():
    """Display recent transactions with enhanced formatting"""
    with st.expander("📋 Recent Transactions", expanded=False):
        transactions = load_transactions()
        
        if not transactions.empty:
            recent = transactions.sort_values('date', ascending=False).head(10)
            
            # Create a more structured display
            for _, tx in recent.iterrows():
                amount_color = "green" if tx['amount'] > 0 else "red"
                amount_symbol = "+" if tx['amount'] > 0 else ""
                
                # Create columns for better layout
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                
                with col1:
                    st.write(f"**{pd.to_datetime(tx['date']).strftime('%Y-%m-%d')}**")
                with col2:
                    st.markdown(f":{amount_color}[{amount_symbol}{tx['amount']:.2f} {tx['currency']}]")
                with col3:
                    st.write(f"🏷️ {tx['type'].title()}")
                with col4:
                    st.write(f"📂 {tx['category']}")
                with col5:
                    desc = tx['description'] if pd.notna(tx['description']) and tx['description'] else 'No description'
                    st.write(f"📝 {desc[:20]}{'...' if len(str(desc)) > 20 else ''}")
                
                st.markdown("---")
        else:
            st.info("No transactions yet. Add your first transaction!")



def render_sidebar_summary():
    """Render enhanced sidebar with AI-powered financial summary"""
    with st.sidebar:
        st.header("📊 AI Financial Dashboard")
        
        # Get comprehensive summary
        summary = get_financial_summary()
        
        if summary["total_income"] > 0 or summary["total_expenses"] > 0:
            # Overall metrics
            st.subheader("💰 Overall Summary")
            st.metric("Total Income", f"{summary['total_income']:.2f}")
            st.metric("Total Expenses", f"{summary['total_expenses']:.2f}")
            
            balance_delta = "normal" if summary['balance'] >= 0 else "inverse"
            st.metric("Balance", f"{summary['balance']:.2f}", delta_color=balance_delta)
            
            # Monthly metrics
            st.subheader("📅 This Month")
            st.metric("Monthly Income", f"{summary['monthly_income']:.2f}")
            st.metric("Monthly Expenses", f"{summary['monthly_expenses']:.2f}")
            
            # AI Insights
            st.subheader("🧠 AI Insights")
            st.info(f"📈 {summary['recent_trend']}")
            
            # Top spending categories
            if summary['top_categories']:
                st.subheader("🔥 Top Spending")
                for category, amount in list(summary['top_categories'].items())[:3]:
                    st.write(f"• **{category}**: {amount:.2f}")
            
            # Financial Health Score
            if summary['monthly_expenses'] > 0:
                # Prevent divide by zero error by checking if monthly_income is greater than zero
                if summary['monthly_income'] > 0:
                    savings_rate = (summary['monthly_income'] - summary['monthly_expenses']) / summary['monthly_income'] * 100
                    if savings_rate > 20:
                        health_status = "🟢 Excellent"
                    elif savings_rate > 10:
                        health_status = "🟡 Good"
                    elif savings_rate > 0:
                        health_status = "🟠 Fair"
                    else:
                        health_status = "🔴 Needs Attention"
                    
                    st.subheader("💊 Financial Health")
                    st.write(f"**Status**: {health_status}")
                    st.write(f"**Savings Rate**: {savings_rate:.1f}%")
                else:
                    # Handle case when income is zero but expenses exist
                    st.subheader("💊 Financial Health")
                    st.write(f"**Status**: 🔴 Needs Attention")
                    st.write(f"**Note**: No income recorded this month")
            
            # Quick actions
            st.subheader("⚡ Quick Actions")
            if st.button("🎯 Set Goals", use_container_width=True):
                st.session_state.show_goals = True
            if st.button("📊 View Analytics", use_container_width=True):
                st.session_state.show_analytics = True
                
        else:
            st.info("💡 Start adding transactions to see your AI-powered financial insights!")
        
        # Goals section
        if hasattr(st.session_state, 'show_goals') and st.session_state.show_goals:
            render_goals_section()

def render_goals_section():
    """Render financial goals management section"""
    st.subheader("🎯 Financial Goals")
    
    goals = st.session_state.goals
    
    # Monthly budget goal
    monthly_budget = st.number_input(
        "Monthly Budget (KZT)", 
        value=goals.get("monthly_budget", 100000),
        step=1000
    )
    
    # Savings target
    savings_target = st.number_input(
        "Monthly Savings Target (KZT)", 
        value=goals.get("savings_target", 50000),
        step=1000
    )
    
    # Category limits
    st.write("**Category Spending Limits**")
    category_limits = goals.get("category_limits", {})
    
    for category in ["Food", "Transport", "Entertainment", "Shopping"]:
        limit = st.number_input(
            f"{category} Limit (KZT)",
            value=category_limits.get(category, 10000),
            step=1000,
            key=f"limit_{category}"
        )
        category_limits[category] = limit
    
    # Save goals
    if st.button("💾 Save Goals"):
        updated_goals = {
            "monthly_budget": monthly_budget,
            "savings_target": savings_target,
            "category_limits": category_limits
        }
        st.session_state.goals = updated_goals
        from core.data_manager import save_goals
        save_goals(updated_goals)
        st.success("✅ Goals saved!")
        st.session_state.show_goals = False
        st.rerun()