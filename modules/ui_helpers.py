
# modules/ui_helpers.py

import streamlit as st

# --- Notifications ---
def show_success_message(message):
    """Displays a success message."""
    st.success(f"✓ {message}")

def show_warning_message(message):
    """Displays a warning message."""
    st.warning(f"⚠️ {message}")

def show_error_message(message):
    """Displays an error message."""
    st.error(f"❌ {message}")

# --- Progress Bars and Spinners ---
def show_progress_bar(progress_value, text="Processing..."):
    """Displays a progress bar with optional text."""
    st.progress(progress_value, text=text)

def show_spinner(text="Loading..."):
    """Displays a spinner with text."""
    return st.spinner(text)

# --- Modals/Dialogs (Simulated with expanders or placeholders) ---
def confirm_action_dialog(message, action_callback, key_suffix=""):
    """
    Simulates a confirmation dialog.
    Streamlit doesn't have native modals that block execution easily.
    This uses buttons and session state to manage confirmation.
    """
    st.warning(message)
    col1, col2 = st.columns(2)
    if col1.button("Confirm", key=f"confirm_{key_suffix}"):
        action_callback()
        st.session_state[f"confirm_dialog_{key_suffix}"] = False # Close dialog
        st.rerun() # Rerun to reflect state change
    if col2.button("Cancel", key=f"cancel_{key_suffix}"):
        st.session_state[f"confirm_dialog_{key_suffix}"] = False # Close dialog
        st.rerun()

# --- Tables ---
def display_interactive_table(df, key="interactive_table"):
    """
    Displays an interactive table (using st.data_editor or st.dataframe).
    `st.data_editor` is good for editing.
    """
    if df is not None and not df.empty:
        return st.data_editor(df, key=key, num_rows="dynamic")
    else:
        st.info("No data to display.")
        return None

# --- Tooltips (Streamlit handles tooltips with the `help` parameter on most widgets) ---
# Example: st.text_input("Reddit API Key", help="Enter your Reddit Client ID")