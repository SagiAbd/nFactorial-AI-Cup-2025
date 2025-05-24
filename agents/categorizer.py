"""
Agent for automatically categorizing financial transactions.
"""
import re

def categorize_transaction(description, amount):
    """
    Categorizes a transaction based on its description and amount.
    
    Args:
        description (str): The transaction description or merchant name
        amount (float): The transaction amount
        
    Returns:
        str: The category assigned to this transaction
    """
    if not description:
        return "Uncategorized"
    
    # Convert description to lowercase for case-insensitive matching
    desc_lower = str(description).lower()
    
    # Income patterns
    if amount > 0:
        if any(keyword in desc_lower for keyword in ['salary', 'зарплата', 'payroll', 'wage']):
            return "Salary"
        elif any(keyword in desc_lower for keyword in ['dividend', 'дивиденд']):
            return "Investment"
        elif any(keyword in desc_lower for keyword in ['freelance', 'фриланс', 'contract']):
            return "Freelance"
        elif any(keyword in desc_lower for keyword in ['refund', 'возврат', 'return']):
            return "Refund"
        elif any(keyword in desc_lower for keyword in ['gift', 'подарок']):
            return "Gift"
        elif any(keyword in desc_lower for keyword in ['bonus', 'бонус']):
            return "Bonus"
        else:
            return "Income"
    
    # Expense patterns
    if any(keyword in desc_lower for keyword in ['restaurant', 'кафе', 'cafe', 'ресторан', 'кофе', 'coffee', 'food', 'еда']):
        return "Food"
    elif any(keyword in desc_lower for keyword in ['grocery', 'супермаркет', 'магазин', 'market', 'store']):
        return "Food"
    elif any(keyword in desc_lower for keyword in ['taxi', 'такси', 'uber', 'яндекс', 'yandex', 'indriver', 'bolt', 'car', 'bus', 'transport']):
        return "Transport"
    elif any(keyword in desc_lower for keyword in ['mobile', 'phone', 'телефон', 'internet', 'интернет', 'tv', 'netflix', 'spotify']):
        return "Utilities"
    elif any(keyword in desc_lower for keyword in ['mall', 'shop', 'магазин', 'тц', 'clothing', 'одежда', 'fashion', 'shoes']):
        return "Shopping"
    elif any(keyword in desc_lower for keyword in ['movie', 'кино', 'theatre', 'театр', 'concert', 'концерт', 'entertainment', 'game']):
        return "Entertainment"
    elif any(keyword in desc_lower for keyword in ['doctor', 'hospital', 'врач', 'больница', 'clinic', 'pharmacy', 'аптека']):
        return "Health"
    elif any(keyword in desc_lower for keyword in ['tuition', 'course', 'education', 'school', 'university', 'book']):
        return "Education"
    elif any(keyword in desc_lower for keyword in ['rent', 'аренда', 'mortgage', 'ипотека', 'house', 'квартира', 'apartment']):
        return "Housing"
    
    # Default category if no pattern matches
    return "Uncategorized" 