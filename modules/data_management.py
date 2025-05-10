# modules/data_management.py

import streamlit as st
import pandas as pd
from presidio_analyzer import AnalyzerEngine #, RecognizerRegistry (if customizing)
# from presidio_analyzer.recognizer_registry import RecognizerRegistry # Example if you want to customize
# from presidio_pii_recognizers import CreditCardRecognizer # Example
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from . import ui_helpers, utils
import os
import json # For saving/loading data if not using CSV/Parquet

logger = utils.setup_logger(__name__)

# --- Presidio Setup (Lazy initialization inside functions) ---
@st.cache_resource # Cache the Presidio engines for performance
def get_presidio_analyzer_instance():
    """Returns a cached instance of Presidio AnalyzerEngine."""
    # Configure to load only necessary recognizers for better performance if needed
    # Example:
    # recognizer_registry = RecognizerRegistry()
    # recognizer_registry.load_predefined_recognizers()
    # # Add or remove recognizers here if needed
    # # recognizer_registry.add_recognizer(CreditCardRecognizer(supported_language="en"))
    # analyzer = AnalyzerEngine(registry=recognizer_registry, supported_languages=["en"])
    logger.info("Initializing Presidio AnalyzerEngine instance.")
    return AnalyzerEngine()

@st.cache_resource
def get_presidio_anonymizer_instance():
    """Returns a cached instance of Presidio AnonymizerEngine."""
    logger.info("Initializing Presidio AnonymizerEngine instance.")
    return AnonymizerEngine()

# --- Data Storage and Retrieval (Project Specific) ---
def save_data_to_project(df, data_filename="reddit_data.csv"):
    """Saves DataFrame to the project's path."""
    if 'project_path' not in st.session_state or not st.session_state.project_path:
        ui_helpers.show_error_message("Project path not set. Cannot save data.")
        return False

    project_path = st.session_state.project_path # This should be the folder path
    file_path = os.path.join(project_path, data_filename)

    try:
        # Using CSV for simplicity, Parquet or JSON Lines could be alternatives
        df.to_csv(file_path, index=False)
        # ui_helpers.show_success_message(f"Data saved successfully to {file_path}") # Can be too noisy
        logger.info(f"Data saved to {file_path}")
        return True
    except Exception as e:
        ui_helpers.show_error_message(f"Error saving data to '{file_path}': {e}")
        logger.error(f"Error saving data to {file_path}: {e}")
        return False

def load_data_from_project(data_filename="reddit_data.csv"):
    """Loads DataFrame from the project's path."""
    if 'project_path' not in st.session_state or not st.session_state.project_path:
        return None

    project_path = st.session_state.project_path
    file_path = os.path.join(project_path, data_filename)

    if not os.path.exists(file_path):
        return None

    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        ui_helpers.show_error_message(f"Error loading data from '{file_path}': {e}")
        logger.error(f"Error loading data from {file_path}: {e}")
        return None

# --- Data Redaction ---
def redact_text_column(df, column_name):
    """
    Redacts sensitive information from a specified text column in the DataFrame.
    Uses Microsoft Presidio.
    """
    analyzer = get_presidio_analyzer_instance() # Get instance when needed
    anonymizer = get_presidio_anonymizer_instance() # Get instance when needed

    if column_name not in df.columns:
        ui_helpers.show_error_message(f"Column '{column_name}' not found in data.")
        return df, 0 # Return original df and 0 redactions

    redacted_texts = []
    total_redactions_found = 0 # More accurate naming

    # Define anonymization strategy, e.g., replace with <ENTITY_TYPE>
    operators = {
        "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
        "PHONE_NUMBER": OperatorConfig("mask", {"type": "mask", "masking_char": "*", "chars_to_mask": 12, "from_end": False}),
        "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
        "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
        "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
        # Add more entity-specific operators if needed
    }

    progress_bar = st.progress(0, text=f"Redacting data in column '{column_name}'...")
    df_length = len(df)

    for i, text in enumerate(df[column_name].astype(str)): # Ensure text is string
        if pd.isna(text) or not text.strip():
            redacted_texts.append(text) # Keep NaN or empty as is
            if df_length > 0:
                progress_bar.progress((i + 1) / df_length, text=f"Processing row {i+1}/{df_length}...")
            continue
        try:
            # Analyze text to find PII
            analyzer_results = analyzer.analyze(text=text, language='en')

            # Anonymize text using the results from the analyzer
            anonymized_result = anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operators
            )
            redacted_texts.append(anonymized_result.text)
            if len(analyzer_results) > 0:
                total_redactions_found += len(analyzer_results)
        except Exception as e:
            logger.error(f"Error during redaction of row {i} for column '{column_name}': {e}")
            redacted_texts.append(text) # Keep original text on error for that row

        if df_length > 0:
            progress_bar.progress((i + 1) / df_length, text=f"Processing row {i+1}/{df_length}...")

    progress_bar.empty()

    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning issues
    new_column_name = column_name + '_redacted'
    df_copy[new_column_name] = redacted_texts

    # No direct UI message here as per earlier decision, app.py will handle it
    # if total_redactions_found > 0:
    #     ui_helpers.show_success_message(f"Sensitive data redacted. {total_redactions_found} items processed. New column '{new_column_name}' created.")
    # else:
    #     ui_helpers.show_warning_message(f"No sensitive data found by Presidio for redaction in '{column_name}'. New column '{new_column_name}' created.")

    return df_copy, total_redactions_found


# --- Data Editing, Filtering, etc. ---
# These functionalities will often be directly implemented in app.py using Streamlit's
# interactive widgets (like st.data_editor for editing, multiselect for filtering).

def add_codes_to_data(df, codes_list, new_code_column='ai_codes'):
    """
    Adds a list of codes to the DataFrame. Assumes codes_list matches df row count.
    codes_list: A list of lists, where each inner list contains codes for a row.
    """
    if len(df) != len(codes_list):
        ui_helpers.show_error_message("Mismatch between data rows and generated codes count.")
        logger.error(f"Data rows ({len(df)}) and codes count ({len(codes_list)}) mismatch.")
        return df # Return original df

    df_copy = df.copy()
    df_copy[new_code_column] = codes_list
    return df_copy

def save_view(df_view, view_name):
    """Saves a filtered/grouped DataFrame view to the project."""
    if 'project_path' not in st.session_state or not st.session_state.project_path:
        ui_helpers.show_error_message("Project path not set. Cannot save view.")
        return False

    project_path = st.session_state.project_path
    views_subfolder = "views"
    views_dir_path = os.path.join(project_path, views_subfolder)

    # Sanitize view_name for filename
    safe_view_name = "".join(c if c.isalnum() else "_" for c in view_name)
    view_filename = f"view_{safe_view_name.lower()}.csv"
    file_path = os.path.join(views_dir_path, view_filename)

    try:
        os.makedirs(views_dir_path, exist_ok=True)
        df_view.to_csv(file_path, index=False)
        ui_helpers.show_success_message(f"View '{view_name}' saved successfully to project 'views' folder.")
        logger.info(f"View '{view_name}' saved to {file_path}")
        return True
    except Exception as e:
        ui_helpers.show_error_message(f"Error saving view '{view_name}': {e}")
        logger.error(f"Error saving view '{view_name}' to {file_path}: {e}")
        return False

def load_saved_views():
    """Lists and allows loading of saved views."""
    if 'project_path' not in st.session_state or not st.session_state.project_path:
        return {} # No project path, so no views to load

    project_path = st.session_state.project_path
    views_subfolder = "views"
    views_dir_path = os.path.join(project_path, views_subfolder)

    if not os.path.exists(views_dir_path):
        return {} # Views folder doesn't exist

    saved_views = {}
    try:
        for filename in os.listdir(views_dir_path):
            if filename.startswith("view_") and filename.endswith(".csv"):
                # Attempt to reconstruct a more readable view name
                # from: view_my_awesome_view.csv -> My Awesome View
                name_part = filename.replace("view_", "", 1).replace(".csv", "")
                view_name = " ".join(word.capitalize() for word in name_part.split("_"))
                saved_views[view_name] = os.path.join(views_dir_path, filename)
    except Exception as e:
        logger.error(f"Error listing saved views from '{views_dir_path}': {e}")
        ui_helpers.show_error_message(f"Could not read saved views: {e}")
        return {}
    return saved_views