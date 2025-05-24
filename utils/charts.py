"""
Visualization utilities for creating charts and graphs.
"""
import plotly.express as px
import pandas as pd

def create_spending_pie_chart(category_data):
    """
    Creates a pie chart showing spending by category.
    
    Args:
        category_data (dict): Dictionary mapping categories to amounts
        
    Returns:
        fig: A plotly or matplotlib figure object
    """
    # Placeholder for chart creation logic
    return None

def create_spending_trend_chart(transaction_data):
    """
    Creates a line chart showing spending trends over time.
    
    Args:
        transaction_data (pd.DataFrame): Transaction data with dates
        
    Returns:
        fig: A plotly or matplotlib figure object
    """
    # Placeholder for trend chart creation logic
    return None 

def create_category_distribution(transactions):
    """
    Creates a pie chart showing spending distribution by category.
    
    Args:
        transactions (pd.DataFrame): Transaction data
        
    Returns:
        fig: A plotly figure object
    """
    if transactions.empty:
        # Create an empty chart with a message
        data = pd.DataFrame({'Category': ['No Data'], 'Amount': [1]})
        fig = px.pie(data, values='Amount', names='Category', 
                    title='No spending data available')
        return fig
    
    # Filter to expenses only
    expenses = transactions[transactions['amount'] < 0].copy()
    
    if expenses.empty:
        # Create an empty chart with a message
        data = pd.DataFrame({'Category': ['No Expenses'], 'Amount': [1]})
        fig = px.pie(data, values='Amount', names='Category', 
                    title='No expense data available')
        return fig
        
    # Group by category and sum amounts
    expenses['amount'] = expenses['amount'].abs()  # Convert to positive for display
    category_totals = expenses.groupby('category')['amount'].sum().reset_index()
    
    # Create pie chart
    fig = px.pie(
        category_totals, 
        values='amount', 
        names='category',
        hole=0.4,  # Creates a donut chart
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    # Update layout
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=300,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig 