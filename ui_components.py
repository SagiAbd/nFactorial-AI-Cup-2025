import streamlit as st
from datetime import date
import pandas as pd
from data_manager import load_transactions, add_category, get_financial_summary


def render_transaction_type_selector():
    """Render transaction type selection with proper alignment"""
    st.write("**Choose Transaction Type or Import**")
    
    # Using columns with equal proportions for proper alignment
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
            st.rerun()
    
    with col3:
        uploaded_file = st.file_uploader(
            "📁 Import Bank Statement",
            type=["csv", "xlsx", "pdf"],
            help="Upload your bank statement",
            label_visibility="collapsed"
        )
        if uploaded_file:
            st.success(f"📁 {uploaded_file.name}")
            st.info("🚧 AI processing feature coming soon!")


def render_currency_selector():
    """Render currency selection using button selector style"""
    st.write("**Currency**")
    currencies = st.session_state.config["currencies"]
    
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
    
    # Date input
    transaction_date = st.date_input(
        "**Date**", 
        value=date.today(),
        key=f"date_{transaction_type}"
    )
    
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
                    "transaction_date": transaction_date,
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
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                
                with col1:
                    st.write(f"**{tx['date'].strftime('%Y-%m-%d')}**")
                with col2:
                    st.write(f"📂 {tx['category']}")
                with col3:
                    st.markdown(f":{amount_color}[{amount_symbol}{tx['amount']:.2f} {tx['currency']}]")
                with col4:
                    desc = tx['description'] if pd.notna(tx['description']) and tx['description'] else 'No description'
                    st.write(f"📝 {desc[:30]}{'...' if len(str(desc)) > 30 else ''}")
                
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
        from data_manager import save_goals
        save_goals(updated_goals)
        st.success("✅ Goals saved!")
        st.session_state.show_goals = False
        st.rerun()