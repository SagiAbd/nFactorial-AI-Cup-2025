"""
Finance Agent App - Main Streamlit Application
"""
import streamlit as st

st.set_page_config(
    page_title="Finance Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Finance Agent 💰")
st.subheader("Your Personal Financial Assistant")

st.markdown("""
Welcome to Finance Agent! This application helps you:
- Analyze and categorize your financial transactions
- Track your spending habits
- Set and monitor budget goals
- Get personalized financial insights

**Get started by navigating to the Dashboard in the sidebar.**
""")

# Sidebar navigation is handled by Streamlit's multipage app feature
# through the pages/ directory
