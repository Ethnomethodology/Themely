# modules/utils.py
import yaml
import logging
import streamlit as st
import os
import re # For more advanced sanitization

# --- Other Utilities (like generate_project_id) first if needed by config functions ---
def generate_project_id(project_name_str):
    """
    Generates a simple, filesystem-friendly ID from a project name string.
    Used for default config filenames.
    """
    if not project_name_str: return "default_project"
    sanitized = str(project_name_str).strip().lower()
    sanitized = re.sub(r'\s+', '_', sanitized) # Replace spaces with underscores
    sanitized = re.sub(r'[^\w_.-]', '', sanitized) # Remove unwanted characters, allow _ . -
    sanitized = sanitized[:50] # Truncate to a reasonable length for filenames
    while '__' in sanitized: sanitized = sanitized.replace('__', '_')
    sanitized = sanitized.strip('_.- ')
    return sanitized if sanitized else "project"

def sanitize_for_filename(text_str, max_length=30):
    """
    Sanitizes a string to be more suitable for use in a filename,
    especially for query parts. More aggressive than generate_project_id.
    """
    if not text_str: return "no_query"
    sanitized = str(text_str).strip().lower()
    sanitized = re.sub(r'\s+', '_', sanitized)
    sanitized = re.sub(r'[^\w-]', '', sanitized) # Allow only word characters and hyphens
    sanitized = sanitized[:max_length]
    sanitized = sanitized.strip('_')
    return sanitized if sanitized else "query"


# --- Configuration file naming template ---
PROJECT_CONFIG_FILE_TEMPLATE = "{project_name_id}_config.yaml"

# --- Logging Setup ---
def setup_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

module_logger = setup_logger(__name__)

# --- Global Application Configuration ---
def load_app_config(config_filename='config.yaml'):
    try:
        with open(config_filename, 'r') as f:
            config = yaml.safe_load(f)
        return config if config else {}
    except FileNotFoundError: return {}
    except yaml.YAMLError as e:
        st.error(f"Error loading global application configuration '{config_filename}': {e}")
        return {}

def save_app_config(config_data, config_filename='config.yaml'):
    try:
        with open(config_filename, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    except IOError as e: st.error(f"Error saving global application configuration to '{config_filename}': {e}")


# --- Project-Specific Configuration Management ---
def load_project_config(config_file_directory, project_name_user_given):
    if not project_name_user_given:
        module_logger.error("User-given project name is empty, cannot load project config.")
        return {}
    if not config_file_directory or not os.path.isdir(config_file_directory):
        module_logger.warning(f"Config file directory '{config_file_directory}' is invalid or does not exist for project '{project_name_user_given}'.")
        return {}

    project_name_id_for_file = generate_project_id(project_name_user_given)
    config_filename = PROJECT_CONFIG_FILE_TEMPLATE.format(project_name_id=project_name_id_for_file)
    config_path = os.path.join(config_file_directory, config_filename)

    try:
        with open(config_path, 'r') as f:
            project_config = yaml.safe_load(f)
        if project_config:
            project_config.setdefault('project_name', project_name_user_given)
            project_config.setdefault('project_config_file_directory', config_file_directory)
            project_config.setdefault('project_id_for_filename', project_name_id_for_file)
        return project_config if project_config else {}
    except FileNotFoundError:
        module_logger.info(f"Project configuration file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error parsing project configuration from '{config_path}': {e}")
        return {}
    except Exception as e:
        st.error(f"An unexpected error occurred while loading project config '{config_path}': {e}")
        return {}

def save_project_config(config_file_directory, project_name_user_given, project_config_data):
    if not project_name_user_given:
        st.error("Project name (user-given) cannot be empty when saving project configuration.")
        return False
    if not config_file_directory or not os.path.isabs(config_file_directory):
        st.error(f"Configuration file directory must be an absolute path: '{config_file_directory}'")
        return False

    try:
        if not os.path.exists(config_file_directory):
            os.makedirs(config_file_directory, exist_ok=True)
            module_logger.info(f"Created directory for project configuration: {config_file_directory}")
    except OSError as e:
        st.error(f"Could not create directory '{config_file_directory}' for project configuration: {e}")
        return False

    project_config_data['project_name'] = project_name_user_given
    project_config_data['project_config_file_directory'] = config_file_directory
    project_name_id_for_file = generate_project_id(project_name_user_given)
    project_config_data['project_id_for_filename'] = project_name_id_for_file

    config_filename = PROJECT_CONFIG_FILE_TEMPLATE.format(project_name_id=project_name_id_for_file)
    config_path = os.path.join(config_file_directory, config_filename)

    try:
        with open(config_path, 'w') as f:
            yaml.dump(project_config_data, f, default_flow_style=False, sort_keys=False)
        return True
    except IOError as e:
        st.error(f"Error saving project configuration to '{config_path}': {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while saving project config to '{config_path}': {e}")
        return False

class AppError(Exception): pass

def handle_exception(e, context="General Error", logger_instance=None):
    error_message = f"An error occurred: {context} - {str(e)}"
    log_target = logger_instance if logger_instance else module_logger
    log_target.error(error_message, exc_info=True)
    try:
        if st._is_running_with_streamlit: # Check if Streamlit context is available
            st.error(error_message)
    except Exception: pass