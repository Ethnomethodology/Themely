# modules/data_management.py
import streamlit as st
import pandas as pd
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from . import ui_helpers, utils
import os
import json
import glob
from datetime import datetime

logger = utils.setup_logger(__name__)

@st.cache_resource
def get_presidio_analyzer_instance():
    logger.info("Initializing Presidio AnalyzerEngine instance.")
    return AnalyzerEngine()

@st.cache_resource
def get_presidio_anonymizer_instance():
    logger.info("Initializing Presidio AnonymizerEngine instance.")
    return AnonymizerEngine()

DOWNLOADS_SUBFOLDER = "reddit_downloads"
VIEWS_SUBFOLDER = "project_views"
CODEBOOK_SUBFOLDER = "codes" # New constant for codebook
CODEBOOK_FILENAME = "project_codebook.csv" # New constant for codebook filename
CODEBOOK_COLUMNS = ["Code Name", "Description", "Rationale", "Example_ids"] # New constant

def save_downloaded_reddit_data(df, subreddit_name, query_str, fetch_params, timestamp):
    project_path = st.session_state.project_path
    if not project_path:
        ui_helpers.show_error_message("Project path not set. Cannot save downloaded Reddit data.")
        return None
    downloads_dir = os.path.join(project_path, DOWNLOADS_SUBFOLDER)
    try: os.makedirs(downloads_dir, exist_ok=True)
    except OSError as e:
        ui_helpers.show_error_message(f"Could not create downloads directory '{downloads_dir}': {e}")
        return None
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
    sub_sanitized = utils.sanitize_for_filename(subreddit_name, 20)
    query_sanitized = utils.sanitize_for_filename(query_str if query_str else "all_posts", 30)
    filename = f"reddit_data_{sub_sanitized}_{query_sanitized}_{ts_str}.csv"
    file_path = os.path.join(downloads_dir, filename)
    try:
        df.to_csv(file_path, index=False)
        logger.info(f"Saved downloaded Reddit data to: {file_path}")
        return file_path
    except Exception as e:
        ui_helpers.show_error_message(f"Error saving downloaded Reddit data to '{file_path}': {e}")
        return None

def list_downloaded_files_metadata():
    project_path = st.session_state.project_path
    if not project_path: return []
    downloads_dir = os.path.join(project_path, DOWNLOADS_SUBFOLDER)
    if not os.path.isdir(downloads_dir): return []
    all_files_metadata = []
    for filename in os.listdir(downloads_dir):
        if filename.startswith("reddit_data_") and filename.endswith(".csv"):
            file_path = os.path.join(downloads_dir, filename)
            try:
                parts = filename.replace("reddit_data_", "").replace(".csv", "").split('_')
                metadata = {
                    "filename": filename, "filepath": file_path,
                    "subreddit": parts[0] if len(parts) > 0 else "N/A",
                    "query_used": parts[1] if len(parts) > 1 else "N/A",
                    "download_timestamp_str": f"{parts[-2]}_{parts[-1]}" if len(parts) >= 3 else "N/A",
                    "download_datetime": datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S") if len(parts) >=3 else datetime.min,
                    "fetch_params_placeholder": "Params not stored with this version" # Placeholder
                }
                all_files_metadata.append(metadata)
            except Exception as e:
                logger.warning(f"Could not parse metadata from filename '{filename}': {e}")
                all_files_metadata.append({"filename": filename, "filepath": file_path, "subreddit": "ParseError", "download_datetime": datetime.min})
    all_files_metadata.sort(key=lambda x: x.get("download_datetime", datetime.min), reverse=True)
    return all_files_metadata

def load_data_from_specific_file(file_path):
    if not file_path or not os.path.isfile(file_path):
        logger.warning(f"File not found or path is invalid: {file_path}")
        return None
    try: return pd.read_csv(file_path)
    except Exception as e:
        ui_helpers.show_error_message(f"Error loading data from '{file_path}': {e}")
        return None

def save_project_view(df_view, view_name, source_filenames_info=None):
    project_path = st.session_state.project_path
    if not project_path:
        ui_helpers.show_error_message("Project path not set. Cannot save view.")
        return False
    views_dir = os.path.join(project_path, VIEWS_SUBFOLDER)
    try: os.makedirs(views_dir, exist_ok=True)
    except OSError as e:
        ui_helpers.show_error_message(f"Could not create views directory '{views_dir}': {e}")
        return False
    safe_view_name = utils.sanitize_for_filename(view_name, 50)
    view_filename_csv = f"{safe_view_name}.csv"
    view_filepath_csv = os.path.join(views_dir, view_filename_csv)
    view_filename_meta = f"{safe_view_name}_meta.json"
    view_filepath_meta = os.path.join(views_dir, view_filename_meta)
    try:
        df_view.to_csv(view_filepath_csv, index=False)
        view_metadata = {
            "view_name": view_name, "csv_filename": view_filename_csv,
            "creation_timestamp": datetime.now().isoformat(),
            "source_files_info": source_filenames_info if source_filenames_info else "Not specified"
        }
        with open(view_filepath_meta, 'w') as f_meta: json.dump(view_metadata, f_meta, indent=4)
        logger.info(f"View '{view_name}' (CSV and Meta) saved to {views_dir}")
        return True
    except Exception as e:
        ui_helpers.show_error_message(f"Error saving view '{view_name}': {e}")
        return False

def list_created_views_metadata():
    project_path = st.session_state.project_path
    if not project_path: return []
    views_dir = os.path.join(project_path, VIEWS_SUBFOLDER)
    if not os.path.isdir(views_dir): return []
    all_views_metadata = []
    for filename in os.listdir(views_dir):
        if filename.endswith("_meta.json"):
            meta_filepath = os.path.join(views_dir, filename)
            try:
                with open(meta_filepath, 'r') as f_meta: meta_content = json.load(f_meta)
                meta_content["csv_filepath"] = os.path.join(views_dir, meta_content.get("csv_filename", ""))
                all_views_metadata.append(meta_content)
            except Exception as e: logger.warning(f"Could not parse view metadata from '{filename}': {e}")
    all_views_metadata.sort(key=lambda x: x.get("creation_timestamp") or "", reverse=True)
    return all_views_metadata

def save_coded_data_to_view(df_coded_view, view_csv_filepath):
    if not view_csv_filepath or not os.path.isabs(view_csv_filepath):
        ui_helpers.show_error_message(f"Invalid file path for saving coded view: {view_csv_filepath}")
        return False
    view_dir = os.path.dirname(view_csv_filepath)
    if not os.path.isdir(view_dir):
        ui_helpers.show_error_message(f"View directory does not exist: {view_dir}")
        return False
    try:
        df_coded_view.to_csv(view_csv_filepath, index=False)
        logger.info(f"Coded data saved to view file: {view_csv_filepath}")
        return True
    except Exception as e:
        ui_helpers.show_error_message(f"Error saving coded data to view '{view_csv_filepath}': {e}")
        return False

def redact_text_column_in_place(df_to_modify, column_name):
    analyzer = get_presidio_analyzer_instance()
    anonymizer = get_presidio_anonymizer_instance()
    if column_name not in df_to_modify.columns:
        ui_helpers.show_error_message(f"Column '{column_name}' not found in data for redaction.")
        return 0
    total_redactions_found = 0
    operators = {
        "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
        "PHONE_NUMBER": OperatorConfig("mask", {"type": "mask", "masking_char": "*", "chars_to_mask": 12, "from_end": False}),
        "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
        "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
        "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
    }
    texts_to_redact = df_to_modify[column_name].astype(str).tolist()
    redacted_texts_list = []
    progress_bar = st.progress(0, text=f"Redacting data in column '{column_name}'...")
    num_rows = len(texts_to_redact)
    for i, text_content in enumerate(texts_to_redact):
        if pd.isna(text_content) or not str(text_content).strip():
            redacted_texts_list.append(text_content)
        else:
            try:
                analyzer_results = analyzer.analyze(text=text_content, language='en')
                anonymized_result = anonymizer.anonymize(text=text_content, analyzer_results=analyzer_results, operators=operators)
                redacted_texts_list.append(anonymized_result.text)
                if len(analyzer_results) > 0: total_redactions_found += len(analyzer_results)
            except Exception as e:
                logger.error(f"Error during redaction of row {i} for column '{column_name}': {e}")
                redacted_texts_list.append(text_content)
        if num_rows > 0: progress_bar.progress((i + 1) / num_rows, text=f"Processing row {i+1}/{num_rows} for redaction...")
    progress_bar.empty()
    df_to_modify[column_name] = redacted_texts_list
    return total_redactions_found

# --- Codebook Functions ---
def get_codebook_filepath(project_path):
    """Helper to get the full path to the codebook CSV file."""
    if not project_path: return None
    codebook_dir = os.path.join(project_path, CODEBOOK_SUBFOLDER)
    return os.path.join(codebook_dir, CODEBOOK_FILENAME)

def load_codebook(project_path):
    """Loads the project codebook from its CSV file."""
    codebook_filepath = get_codebook_filepath(project_path)
    if not codebook_filepath:
        logger.error("Project path not available, cannot load codebook.")
        return pd.DataFrame(columns=CODEBOOK_COLUMNS)
        
    if os.path.exists(codebook_filepath):
        try:
            df = pd.read_csv(codebook_filepath)
            # Ensure all necessary columns exist, even if CSV was manipulated
            for col in CODEBOOK_COLUMNS:
                if col not in df.columns:
                    df[col] = "" if col != "Example_ids" else "" # Default for Example_ids as string
            # Ensure Example_ids is string
            if 'Example_ids' in df.columns:
                 df['Example_ids'] = df['Example_ids'].astype(str).fillna('')
            logger.info(f"Codebook loaded from {codebook_filepath}")
            return df[CODEBOOK_COLUMNS] # Return in defined order
        except Exception as e:
            logger.error(f"Error loading codebook from '{codebook_filepath}': {e}")
            ui_helpers.show_error_message(f"Could not load codebook: {e}. A new one will be started.")
            return pd.DataFrame(columns=CODEBOOK_COLUMNS)
    else:
        logger.info("No existing codebook file found. Starting a new one.")
        return pd.DataFrame(columns=CODEBOOK_COLUMNS)

def save_codebook(df_codebook, project_path):
    """Saves the project codebook to its CSV file."""
    codebook_filepath = get_codebook_filepath(project_path)
    if not codebook_filepath:
        ui_helpers.show_error_message("Project path not set. Cannot save codebook.")
        return False

    codebook_dir = os.path.dirname(codebook_filepath)
    try:
        os.makedirs(codebook_dir, exist_ok=True)
        # Ensure Example_ids is string before saving
        df_to_save = df_codebook.copy()
        if 'Example_ids' in df_to_save.columns:
            df_to_save['Example_ids'] = df_to_save['Example_ids'].astype(str).fillna('')

        df_to_save.to_csv(codebook_filepath, index=False)
        logger.info(f"Codebook saved to {codebook_filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving codebook to '{codebook_filepath}': {e}")
        ui_helpers.show_error_message(f"Error saving codebook: {e}")
        return False

# --- Legacy functions ---
def save_data_to_project(df, data_filename="generic_data.csv"):
    project_path = st.session_state.project_path
    if not project_path: return False
    file_path = os.path.join(project_path, data_filename)
    try: df.to_csv(file_path, index=False); return True
    except Exception: return False

def load_data_from_project(data_filename="generic_data.csv"):
    project_path = st.session_state.project_path
    if not project_path: return None
    file_path = os.path.join(project_path, data_filename)
    if not os.path.exists(file_path): return None
    try: return pd.read_csv(file_path)
    except Exception: return None