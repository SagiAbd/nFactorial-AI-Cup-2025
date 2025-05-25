"""
Currency utilities for FinSight
- Exchange rate management
- Currency conversion
- Currency formatting
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Constants
CURRENCY_SYMBOLS = {
    "KZT": "₸",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "RUB": "₽",
    "CNY": "¥",
    "JPY": "¥"
}

# Path for storing exchange rates
EXCHANGE_RATES_FILE = os.path.join("data", "exchange_rates.json")

def get_exchange_rates():
    """
    Get current exchange rates for KZT and other major currencies.
    
    Returns:
        dict: Dictionary of exchange rates with currency codes as keys
              and conversion rates to KZT as values.
    """
    # Ensure data directory exists
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Check if we have cached rates that are less than 24 hours old
    if os.path.exists(EXCHANGE_RATES_FILE):
        try:
            with open(EXCHANGE_RATES_FILE, 'r') as f:
                data = json.load(f)
                last_updated = datetime.fromisoformat(data['last_updated'])
                
                # If rates are fresh (less than 24 hours old), use them
                if datetime.now() - last_updated < timedelta(hours=24):
                    return data['rates']
        except Exception as e:
            logger.warning(f"Error reading cached exchange rates: {e}")
    
    # Default exchange rates (as of November 2023)
    default_rates = {
        "KZT": 1.0,
        "USD": 470.0,
        "EUR": 505.0,
        "GBP": 585.0,
        "RUB": 5.2,
        "CNY": 65.0,
        "JPY": 3.15
    }
    
    try:
        # Try to get latest rates from an API
        # Using Exchange Rate API (there are many alternatives)
        api_key = os.environ.get("EXCHANGE_RATE_API_KEY", "")
        
        if api_key:
            response = requests.get(f"https://v6.exchangerate-api.com/v6/{api_key}/latest/KZT")
            
            if response.status_code == 200:
                api_data = response.json()
                
                if api_data.get('result') == 'success':
                    # The API returns rates as KZT to other currencies
                    # We need to invert these to get other currencies to KZT
                    rates = {"KZT": 1.0}
                    
                    for currency, rate in api_data['conversion_rates'].items():
                        if currency != 'KZT' and currency in CURRENCY_SYMBOLS:
                            # Invert rate since we want X currency to KZT
                            rates[currency] = 1 / rate
                    
                    # Cache the rates
                    with open(EXCHANGE_RATES_FILE, 'w') as f:
                        json.dump({
                            'last_updated': datetime.now().isoformat(),
                            'rates': rates
                        }, f)
                    
                    return rates
        
        # If API call fails or no API key, use default rates
        return default_rates
        
    except Exception as e:
        logger.error(f"Error fetching exchange rates: {e}")
        return default_rates

def convert_to_kzt(amount, currency):
    """
    Convert an amount from specified currency to KZT
    
    Args:
        amount (float): The amount to convert
        currency (str): The source currency code (e.g., USD, EUR)
        
    Returns:
        float: The amount converted to KZT
    """
    if not amount or pd.isna(amount):
        return 0.0
    
    if currency == "KZT":
        return float(amount)
    
    # Get exchange rates
    rates = get_exchange_rates()
    
    # If currency not in our rates, return original amount
    if currency not in rates:
        logger.warning(f"Unknown currency: {currency}, returning original amount")
        return float(amount)
    
    # Convert to KZT
    return float(amount) * rates[currency]

def format_currency(amount, currency="KZT", include_symbol=True, colorize=False, transaction_type=None):
    """
    Format a currency amount with proper symbol and formatting
    
    Args:
        amount (float): The amount to format
        currency (str): The currency code
        include_symbol (bool): Whether to include the currency symbol
        colorize (bool): Whether to add HTML color based on transaction type
        transaction_type (str): Transaction type (income/expense) for colorizing
        
    Returns:
        str: Formatted currency string
    """
    if amount is None or pd.isna(amount):
        return "-"
    
    amount = float(amount)
    
    # Get symbol if needed
    symbol = ""
    if include_symbol:
        symbol = CURRENCY_SYMBOLS.get(currency, currency)
    
    # Format with thousands separator and 2 decimal places
    formatted = f"{symbol} {abs(amount):,.2f}"
    
    # Add sign and color if needed
    if colorize and transaction_type:
        color = "green" if transaction_type == "income" else "red"
        sign = "+" if transaction_type == "income" else "-"
        return f"<span style='color:{color}'>{sign}{formatted}</span>"
    
    return formatted 