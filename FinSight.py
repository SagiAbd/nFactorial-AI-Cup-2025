import streamlit as st
import pandas as pd
import os
import sys
import tempfile
from datetime import date, datetime
import logging

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
from utils.transaction_processor import TransactionProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path for data directory
DATA_DIR = "data"
TRANSACTIONS_FILE = os.path.join(DATA_DIR, "transactions.csv")

def reset_form_state():
    """Reset form-related session state variables"""
    st.session_state.selected_type = None
    st.session_state.selected_currency = None
    st.session_state.selected_category = None
    st.session_state.form_submitted = False

def reset_save_state():
    """Reset save-related session state variables"""
    st.session_state.save_attempted = False

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary file and return the path"""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            # Write the uploaded file to the temporary file
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving uploaded file: {e}")
        return None

def process_files(uploaded_files, api_key=None):
    """Process uploaded files and extract transactions"""
    if not uploaded_files:
        return pd.DataFrame()
    
    # Create a processor instance
    try:
        processor = TransactionProcessor(api_key=api_key)
    except ValueError as e:
        st.error(f"Error initializing processor: {e}")
        return pd.DataFrame()
    
    # Save files to disk
    saved_files = []
    for file in uploaded_files:
        saved_path = save_uploaded_file(file)
        if saved_path:
            saved_files.append(saved_path)
    
    if not saved_files:
        st.error("No files were successfully saved for processing.")
        return pd.DataFrame()
    
    try:
        # Process the files
        logger.info(f"Processing {len(saved_files)} files")
        result = processor.process_files(saved_files)
        
        # Check if we got any transactions
        if result["error_count"] > 0 and result["transaction_count"] == 0:
            st.error(f"Error processing files: {', '.join(result.get('errors', ['Unknown error']))}")
            return pd.DataFrame()
        
        # Check if we got any transactions
        if not result["transactions"]:
            st.warning("No transactions were found in the uploaded files.")
            return pd.DataFrame()
        
        # Convert to DataFrame
        transactions_df = pd.DataFrame(result["transactions"])
        
        # Set standardized currency to avoid duplicates
        if not transactions_df.empty and 'currency' in transactions_df.columns:
            # Find the most common currency in the dataset
            most_common_currency = transactions_df['currency'].mode()[0]
            logger.info(f"Setting all currencies to {most_common_currency} to avoid duplicates")
            transactions_df['currency'] = most_common_currency
        
        # Display summary
        st.success(f"✅ Successfully extracted {len(transactions_df)} transactions from {len(saved_files)} files!")
        
        return transactions_df
    
    except Exception as e:
        st.error(f"Error processing files: {e}")
        logger.error(f"Error processing files: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def save_transactions_to_file(transactions_df):
    """Save extracted transactions to the transactions.csv file"""
    if transactions_df is None or transactions_df.empty:
        st.error("No transactions to save!")
        return 0, 0
    
    # Log transaction count before saving
    logger.info(f"Attempting to save {len(transactions_df)} transactions")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Track counters for success/errors
    saved_count = 0
    error_count = 0
    
    # Initialize session state for saved transaction IDs if not exists
    if "saved_transaction_ids" not in st.session_state:
        st.session_state.saved_transaction_ids = set()
    
    # Create unique identifiers for each transaction to avoid duplicates
    transactions_df['transaction_id'] = transactions_df.apply(
        lambda row: f"{row['date']}_{row['description']}_{row['amount']}",
        axis=1
    )
    
    try:
        # Check if transactions file exists
        if os.path.exists(TRANSACTIONS_FILE):
            # Read existing transactions
            existing_df = pd.read_csv(TRANSACTIONS_FILE)
            logger.info(f"Found existing transactions file with {len(existing_df)} transactions")
            
            # Check for transaction_id column in existing file
            if 'transaction_id' not in existing_df.columns:
                # Create transaction_id for existing transactions
                existing_df['transaction_id'] = existing_df.apply(
                    lambda row: f"{row['date']}_{row['description']}_{row['amount']}",
                    axis=1
                )
                logger.info("Added transaction_id to existing transactions")
            
            # Remove transactions that are already saved (check by ID)
            new_transactions = transactions_df[~transactions_df['transaction_id'].isin(existing_df['transaction_id'])]
            
            # Skip any transactions we've already saved in this session
            new_transactions = new_transactions[~new_transactions['transaction_id'].isin(st.session_state.saved_transaction_ids)]
            
            logger.info(f"After filtering duplicates, {len(new_transactions)} new transactions to save")
            
            if not new_transactions.empty:
                # For each new transaction, save it individually
                for index, transaction in new_transactions.iterrows():
                    try:
                        # Convert transaction to dict
                        tx_dict = transaction.to_dict()
                        
                        # Remove source and transaction_id fields
                        if 'source' in tx_dict:
                            del tx_dict['source']
                        if 'transaction_id' in tx_dict:
                            tx_id = tx_dict['transaction_id']
                            del tx_dict['transaction_id']
                        else:
                            tx_id = f"{tx_dict['date']}_{tx_dict['description']}_{tx_dict['amount']}"
                        
                        # Ensure amount is numeric
                        if isinstance(tx_dict['amount'], str):
                            # Try to convert to float
                            try:
                                tx_dict['amount'] = float(tx_dict['amount'].replace(',', '.'))
                            except ValueError:
                                logger.warning(f"Could not convert amount to float: {tx_dict['amount']}")
                        
                        # Ensure we use transaction_type instead of type
                        if 'type' in tx_dict:
                            tx_dict['transaction_type'] = tx_dict['type']
                            del tx_dict['type']
                        
                        # Convert date to transaction_date
                        if 'date' in tx_dict:
                            # Convert string date to datetime object
                            try:
                                tx_dict['transaction_date'] = datetime.strptime(tx_dict['date'], '%Y-%m-%d')
                                del tx_dict['date']
                            except ValueError:
                                logger.warning(f"Could not convert date to datetime: {tx_dict['date']}")
                                # Create a default date as fallback
                                tx_dict['transaction_date'] = datetime.now()
                                del tx_dict['date']
                        
                        # Log transaction before saving
                        logger.info(f"Saving transaction: {tx_dict}")
                        
                        # Save transaction
                        save_transaction(**tx_dict)
                        
                        # Add to saved ids
                        st.session_state.saved_transaction_ids.add(tx_id)
                        
                        saved_count += 1
                    except Exception as e:
                        logger.error(f"Error saving transaction: {e}")
                        error_count += 1
                
                # Read updated transactions
                updated_df = pd.read_csv(TRANSACTIONS_FILE)
                logger.info(f"Transactions file now has {len(updated_df)} transactions")
                
                if saved_count > 0:
                    st.success(f"✅ Successfully saved {saved_count} new transactions to {TRANSACTIONS_FILE}")
                if error_count > 0:
                    st.warning(f"⚠️ Failed to save {error_count} transactions due to errors")
            else:
                st.info("No new transactions to save - all transactions already exist in the file.")
        else:
            # First time saving, just save all transactions
            logger.info(f"No existing transactions file. Saving all {len(transactions_df)} transactions")
            
            # Save each transaction individually
            for index, transaction in transactions_df.iterrows():
                try:
                    # Convert transaction to dict
                    tx_dict = transaction.to_dict()
                    
                    # Remove source and transaction_id fields
                    if 'source' in tx_dict:
                        del tx_dict['source']
                    if 'transaction_id' in tx_dict:
                        tx_id = tx_dict['transaction_id']
                        del tx_dict['transaction_id']
                    else:
                        tx_id = f"{tx_dict['date']}_{tx_dict['description']}_{tx_dict['amount']}"
                    
                    # Ensure amount is numeric
                    if isinstance(tx_dict['amount'], str):
                        # Try to convert to float
                        try:
                            tx_dict['amount'] = float(tx_dict['amount'].replace(',', '.'))
                        except ValueError:
                            logger.warning(f"Could not convert amount to float: {tx_dict['amount']}")
                    
                    # Ensure we use transaction_type instead of type
                    if 'type' in tx_dict:
                        tx_dict['transaction_type'] = tx_dict['type']
                        del tx_dict['type']
                    
                    # Convert date to transaction_date
                    if 'date' in tx_dict:
                        # Convert string date to datetime object
                        try:
                            tx_dict['transaction_date'] = datetime.strptime(tx_dict['date'], '%Y-%m-%d')
                            del tx_dict['date']
                        except ValueError:
                            logger.warning(f"Could not convert date to datetime: {tx_dict['date']}")
                            # Create a default date as fallback
                            tx_dict['transaction_date'] = datetime.now()
                            del tx_dict['date']
                    
                    # Log transaction before saving
                    logger.info(f"Saving transaction: {tx_dict}")
                    
                    # Save transaction
                    save_transaction(**tx_dict)
                    
                    # Add to saved ids
                    st.session_state.saved_transaction_ids.add(tx_id)
                    
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Error saving transaction: {e}")
                    error_count += 1
            
            if saved_count > 0:
                st.success(f"✅ Successfully saved {saved_count} new transactions to {TRANSACTIONS_FILE}")
            if error_count > 0:
                st.warning(f"⚠️ Failed to save {error_count} transactions due to errors")
    
    except Exception as e:
        st.error(f"Error saving transactions: {e}")
        logger.error(f"Error saving transactions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0, 1
    
    return saved_count, error_count

def render_document_processing():
    """Render the document processing section"""
    st.markdown("## 📄 Process Bank Documents")
    st.write("Upload your bank statements or receipts to extract and categorize transactions.")
    st.write("Supported formats: CSV, PDF, XLSX, JPG, PNG")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload your bank statements or receipts", 
        accept_multiple_files=True,
        type=["csv", "pdf", "xlsx", "xls", "jpg", "jpeg", "png"],
        key="document_uploader"
    )
    
    # Add processing options
    st.write("### Processing Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        auto_save = st.checkbox(
            "Auto-save transactions",
            value=False,
            help="Automatically save transactions to your data file"
        )
    
    # Add a flag to track if save has been attempted
    if 'save_attempted' not in st.session_state:
        st.session_state.save_attempted = False
    
    # Process button
    if uploaded_files:
        if st.button("Process Documents", type="primary", key="process_button"):
            # Reset save state when processing new documents
            st.session_state.save_attempted = False
            
            with st.spinner("Processing your documents..."):
                # Get API key from secrets
                api_key = None
                if 'OPENAI_API_KEY' in st.secrets:
                    api_key = st.secrets['OPENAI_API_KEY']
                
                # Process files
                processed_df = process_files(uploaded_files, api_key)
                
                if processed_df is not None and not processed_df.empty:
                    # Show a preview of the processed transactions
                    st.success(f"📁 Successfully processed {len(uploaded_files)} document(s)")
                    
                    # Check for unclear data
                    unclear_mask = (
                        processed_df['description'].astype(str).str.contains('[UNCLEAR]', case=False) |
                        processed_df['date'].astype(str).str.contains('[UNCLEAR]', case=False) |
                        processed_df['amount'].astype(str).str.contains('[UNCLEAR]', case=False)
                    )
                    
                    unclear_count = unclear_mask.sum()
                    
                    if unclear_count > 0:
                        st.warning(f"⚠️ Found {unclear_count} transactions with unclear data")
                        
                        # Display unclear transactions
                        with st.expander("View Transactions with Unclear Data", expanded=True):
                            st.dataframe(processed_df[unclear_mask])
                            
                            st.info("These transactions need manual review. You can edit them directly in the transactions page.")
                    
                    # Display all transactions
                    with st.expander("Preview All Processed Transactions", expanded=True):
                        st.dataframe(processed_df)
                        
                        # Download button
                        csv = processed_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Processed Data", 
                            csv, 
                            "processed_transactions.csv", 
                            "text/csv", 
                            key="download-csv"
                        )
                    
                    # Add a clear message about saving options
                    st.markdown("""
                    ### Save Your Transactions
                    You have two options to save these transactions to your database:
                    
                    1. **Auto-save**: Enable the checkbox at the top of the page before processing
                    2. **Manual save**: Use the 'Save All Transactions' button below
                    """)
                    
                    # Auto-save or manual save
                    if auto_save and not st.session_state.save_attempted:
                        with st.spinner("Saving transactions..."):
                            st.session_state.save_attempted = True
                            success_count, error_count = save_transactions_to_file(processed_df)
                            
                            if success_count > 0:
                                st.success(f"✅ Successfully saved {success_count} transactions!")
                                if error_count > 0:
                                    st.warning(f"⚠️ Failed to save {error_count} transactions (unclear data)")
                            else:
                                st.error("❌ Failed to save any transactions")
                    else:
                        # Manual save buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Always show the save button, but disable it if already attempted
                            button_label = "Save All Transactions"
                            if st.session_state.save_attempted:
                                button_label = "Save All Transactions (Already Saved)"
                                
                            if st.button(button_label, type="primary", disabled=st.session_state.save_attempted, key="save_button"):
                                with st.spinner("Saving transactions..."):
                                    st.session_state.save_attempted = True
                                    success_count, error_count = save_transactions_to_file(processed_df)
                                    
                                    if success_count > 0:
                                        st.success(f"✅ Successfully saved {success_count} transactions!")
                                        if error_count > 0:
                                            st.warning(f"⚠️ Failed to save {error_count} transactions (unclear data)")
                                    else:
                                        st.error("❌ Failed to save any transactions")
                        
                        with col2:
                            # Make the reset button more visible
                            if st.session_state.save_attempted:
                                st.info("Click 'Reset' to enable saving again")
                                if st.button("Reset", 
                                            help="Click this to reset the save state and enable saving again",
                                            type="secondary",
                                            key="reset_button"):
                                    reset_save_state()
                                    st.experimental_rerun()
                else:
                    st.error("❌ Failed to process the uploaded documents. Please check the file format.")
    
    # Display help text
    with st.expander("How to use this feature"):
        st.markdown("""
        ### How to Process Bank Documents
        
        This feature allows you to automatically extract transactions from various bank documents:
        
        1. **Upload Files**: Upload your bank statements, receipts, or check images.
        2. **Process**: Click the "Process Documents" button to extract transactions.
        3. **Review**: Check the extracted transactions for accuracy.
        4. **Save**: Save the transactions to your transactions database.
        
        ### Handling Unclear Data
        
        If the system can't clearly read some information, it will mark it with [UNCLEAR].
        You may need to manually edit these transactions after saving.
        
        ### Supported File Types
        
        - **CSV**: Spreadsheet files from your bank's export feature
        - **PDF**: Bank statements, digital receipts
        - **XLSX/XLS**: Excel spreadsheets
        - **JPG/PNG**: Photos of receipts or checks
        
        For best results, ensure your documents are clear and readable.
        """)

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
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Add Transaction", "Process Documents", "Chat Assistant"])
    
    with tab1:
        # Transaction type selection
        render_transaction_type_selector()
        
        # Render appropriate form based on selection
        if st.session_state.selected_type in ["expense", "income"]:
            # Handle form submission
            transaction_data = render_transaction_form(st.session_state.selected_type)
            
            if transaction_data:
                success = save_transaction(**transaction_data)
                if success:
                    st.success(f"✅ {transaction_data['transaction_type'].title()} saved successfully!")
                    reset_form_state()
                    st.rerun()
                else:
                    st.error("❌ Failed to save transaction!")
            
            # Close form button
            if st.button("❌ Close Form", type="secondary"):
                reset_form_state()
                st.rerun()
        else:
            st.info("👆 Select a transaction type above to get started!")
        
        # Recent transactions
        st.markdown("---")
        render_recent_transactions()
    
    with tab2:
        # Document processing section
        render_document_processing()
    
    with tab3:
        # Chat interface
        render_chat_interface()
    
    # Sidebar summary
    render_sidebar_summary()


if __name__ == "__main__":
    main() 