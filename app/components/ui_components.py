import streamlit as st
import pandas as pd
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
from core.data_manager import (
    load_transactions, 
    get_financial_summary, 
    get_category_spending,
    get_monthly_comparison
)
from utils.budget_math import calculate_budget_progress
from utils.charts import create_spending_trend_chart, create_category_distribution


def render_transaction_type_selector():
    """Render transaction type selection buttons"""
    st.subheader("Add New Transaction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💰 Income", key="income_btn", type="primary" if st.session_state.selected_type == "income" else "secondary", use_container_width=True):
            st.session_state.selected_type = "income"
    
    with col2:
        if st.button("💸 Expense", key="expense_btn", type="primary" if st.session_state.selected_type == "expense" else "secondary", use_container_width=True):
            st.session_state.selected_type = "expense"


def render_transaction_form(transaction_type):
    """Render form for adding new transaction"""
    config = st.session_state.config
    
    st.subheader(f"New {'Income' if transaction_type == 'income' else 'Expense'}")
    
    with st.form(key=f"transaction_form_{transaction_type}"):
        # Date picker
        transaction_date = st.date_input(
            "Date",
            value=date.today(),
            key=f"date_{transaction_type}"
        )
        
        # Amount and currency
        col1, col2 = st.columns([2, 1])
        with col1:
            amount = st.number_input(
                "Amount",
                min_value=0.01,
                value=100.0,
                step=10.0,
                key=f"amount_{transaction_type}"
            )
        
        with col2:
            # Set or initialize selected currency
            if st.session_state.selected_currency is None:
                st.session_state.selected_currency = config["currencies"][0]
                
            currency = st.selectbox(
                "Currency",
                options=config["currencies"],
                index=config["currencies"].index(st.session_state.selected_currency),
                key=f"currency_{transaction_type}"
            )
            # Remember the selected currency
            st.session_state.selected_currency = currency
        
        # Category selection
        categories_key = f"{transaction_type}_categories"
        category_names = [cat["name"] for cat in config[categories_key]]
        category_icons = {cat["name"]: cat["icon"] for cat in config[categories_key]}
        
        # Set or initialize selected category
        if st.session_state.selected_category is None or st.session_state.selected_category not in category_names:
            st.session_state.selected_category = category_names[0]
            
        # Display categories as a grid of buttons
        st.write("Category")
        cols = st.columns(4)
        
        for i, name in enumerate(category_names):
            col_idx = i % 4
            with cols[col_idx]:
                icon = category_icons[name]
                if st.button(
                    f"{icon} {name}", 
                    key=f"cat_{name}_{transaction_type}",
                    type="primary" if st.session_state.selected_category == name else "secondary",
                    use_container_width=True
                ):
                    st.session_state.selected_category = name
        
        # Add custom category option
        with st.expander("Add Custom Category"):
            custom_name = st.text_input("Category Name", key=f"custom_name_{transaction_type}")
            custom_icon = st.text_input("Icon (emoji)", key=f"custom_icon_{transaction_type}", placeholder="💼")
            
            if st.button("Add Category", key=f"add_cat_btn_{transaction_type}"):
                from core.data_manager import add_category
                if custom_name and custom_icon:
                    success = add_category(transaction_type, custom_name, custom_icon)
                    if success:
                        st.success(f"Added category {custom_icon} {custom_name}")
                        # Need to rerun to update the category list
                        st.rerun()
                    else:
                        st.error("Category already exists")
        
        # Display the selected category
        st.info(f"Selected Category: {category_icons.get(st.session_state.selected_category, '📋')} {st.session_state.selected_category}")
        
        # Description
        description = st.text_area(
            "Description (optional)",
            key=f"description_{transaction_type}",
            placeholder="Add details about this transaction..."
        )
        
        # Submit button
        submitted = st.form_submit_button(
            f"Save {'Income' if transaction_type == 'income' else 'Expense'}", 
            type="primary",
            use_container_width=True
        )
        
        if submitted:
            st.session_state.form_submitted = True
            
            return {
                "transaction_date": transaction_date,
                "amount": amount,
                "category": st.session_state.selected_category,
                "currency": currency,
                "transaction_type": transaction_type,
                "description": description
            }
    
    return None


def render_recent_transactions():
    """Render table of recent transactions"""
    st.subheader("Recent Transactions")
    
    transactions = load_transactions()
    
    if transactions.empty:
        st.info("No transactions yet. Add some using the form above!")
        return
    
    # Sort by date descending
    transactions = transactions.sort_values(by='date', ascending=False).head(10)
    
    # Format for display
    display_df = transactions.copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
    display_df['amount'] = display_df.apply(
        lambda x: f"{x['amount']:.2f} {x['currency']}", 
        axis=1
    )
    
    # Add emoji indicators for transaction type
    display_df['type'] = display_df['type'].apply(
        lambda x: "💰 Income" if x == "income" else "💸 Expense"
    )
    
    # Reorder columns for display
    display_df = display_df[['date', 'type', 'category', 'amount', 'description']]
    display_df.columns = ['Date', 'Type', 'Category', 'Amount', 'Description']
    
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True
    )


def render_sidebar_summary():
    """Render financial summary in sidebar"""
    with st.sidebar:
        st.title("💰 Financial Summary")
        
        summary = get_financial_summary()
        
        # Display total balance
        st.metric(
            "Current Balance",
            f"{summary['balance']:.2f} KZT",
            delta=f"{summary['monthly_income'] - summary['monthly_expenses']:.2f} this month"
        )
        
        st.markdown("---")
        
        # Monthly income vs expenses
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Income", f"{summary['monthly_income']:.2f}")
        with col2:
            st.metric("Expenses", f"{summary['monthly_expenses']:.2f}")
        
        # Calculate savings rate
        if summary['monthly_income'] > 0:
            savings_rate = (summary['monthly_income'] - summary['monthly_expenses']) / summary['monthly_income'] * 100
            st.progress(max(min(savings_rate / 100, 1), 0), f"Savings Rate: {savings_rate:.1f}%")
        
        st.markdown("---")
        
        # Top spending categories
        st.subheader("Top Spending")
        
        if summary['top_categories']:
            for category, amount in summary['top_categories'].items():
                st.metric(category, f"{amount:.2f} KZT")
                
                # Get goal limit for category if it exists
                goals = st.session_state.goals
                if 'category_limits' in goals and category in goals['category_limits']:
                    limit = goals['category_limits'][category]
                    progress = min(amount / limit, 1) * 100
                    st.progress(progress / 100, f"{progress:.1f}% of {limit} KZT limit")
        else:
            st.info("No expenses recorded yet")
        
        # Recent trend
        st.markdown("---")
        st.subheader("Recent Trend")
        st.write(summary['recent_trend'])
        
        # Budget progress
        st.markdown("---")
        st.subheader("Monthly Budget")
        
        goals = st.session_state.goals
        budget_progress = calculate_budget_progress(summary['monthly_expenses'], goals['monthly_budget'])
        
        # Display progress bar
        st.progress(budget_progress / 100, f"{budget_progress:.1f}% of budget used")
        
        remaining = goals['monthly_budget'] - summary['monthly_expenses']
        st.metric("Remaining Budget", f"{remaining:.2f} KZT")
        
        # Spending trend chart
        st.markdown("---")
        st.subheader("Spending by Category")
        chart = create_category_distribution(transactions=load_transactions())
        st.plotly_chart(chart, use_container_width=True)
        
        # Monthly comparison
        st.markdown("---")
        st.subheader("Monthly Comparison")
        monthly_data = get_monthly_comparison()
        
        if monthly_data:
            # Convert period to string for display
            data = []
            for period, values in monthly_data['amount'].items():
                data.append({
                    'Month': str(period),
                    'Income': values['income'],
                    'Expenses': values['expenses'],
                    'Balance': values['balance']
                })
            
            df = pd.DataFrame(data)
            
            # Create a bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['Month'],
                y=df['Income'],
                name='Income',
                marker_color='green'
            ))
            fig.add_trace(go.Bar(
                x=df['Month'],
                y=df['Expenses'],
                name='Expenses',
                marker_color='red'
            ))
            
            fig.update_layout(
                barmode='group',
                margin=dict(l=0, r=0, t=30, b=0),
                height=300,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for monthly comparison") 