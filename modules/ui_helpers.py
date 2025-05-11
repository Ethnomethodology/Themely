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
    st.progress(progress_value, text=text) # This is a direct st call, placeholder if custom needed

def show_spinner(text="Loading..."):
    """Displays a spinner with text. Use with 'with' statement."""
    return st.spinner(text)

# --- Modals/Dialogs (Simulated with expanders or buttons) ---
def confirm_action_dialog(message, action_callback, key_suffix=""):
    """
    Simulates a confirmation dialog using buttons.
    The calling function needs to manage the state that shows/hides this dialog.
    """
    st.warning(message) # Display the confirmation message
    col1, col2, col3 = st.columns([1,1,3]) # Adjust column ratios as needed
    
    confirmed = col1.button("Confirm", key=f"confirm_dialog_btn_{key_suffix}", type="primary")
    cancelled = col2.button("Cancel", key=f"cancel_dialog_btn_{key_suffix}")

    if confirmed:
        action_callback() # Execute the action
        return "confirmed"
    if cancelled:
        return "cancelled"
    return None # No action taken yet

# --- Tables ---
def display_interactive_table(df, key="interactive_table", height=None):
    """
    Displays an interactive table using st.data_editor.
    `st.data_editor` is good for editing.
    """
    if df is not None and not df.empty:
        return st.data_editor(df, key=key, num_rows="dynamic", height=height, use_container_width=True)
    else:
        # st.info("No data to display in table.") # Potentially too verbose if table is often empty by design
        return pd.DataFrame() # Return empty DF instead of None to avoid errors with subsequent operations