"""
Utilities for budget calculations and financial analysis.
"""

def calculate_monthly_spending(transaction_data):
    """
    Calculates total spending by month.
    
    Args:
        transaction_data (pd.DataFrame): Transaction data with dates
        
    Returns:
        dict: Dictionary mapping months to total spending
    """
    # Placeholder for spending calculation logic
    return {}

def calculate_budget_progress(current_spending, budget_limit):
    """
    Calculates progress towards a budget limit as a percentage.
    
    Args:
        current_spending (float): Current spending amount
        budget_limit (float): Budget limit amount
        
    Returns:
        float: Progress percentage (0-100)
    """
    if budget_limit <= 0:
        return 100.0  # Avoid division by zero
    
    progress = (current_spending / budget_limit) * 100
    return min(progress, 100.0)  # Cap at 100%

def analyze_budget_categories(budget_goals, actual_spending):
    """
    Analyzes budget progress by category.
    
    Args:
        budget_goals (dict): Dictionary mapping categories to budget limits
        actual_spending (dict): Dictionary mapping categories to actual spending
        
    Returns:
        dict: Dictionary with budget progress metrics
    """
    # Categorize spending relative to budget
    on_track = []
    over_budget = []
    
    for category, limit in budget_goals.items():
        spent = actual_spending.get(category, 0)
        if spent <= limit:
            on_track.append(category)
        else:
            over_budget.append(category)
    
    return {
        "on_track": on_track, 
        "over_budget": over_budget
    } 