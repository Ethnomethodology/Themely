# modules/utils.py

import yaml
import logging
import streamlit as st
import os

# Configuration file path
CONFIG_FILE = 'config.yaml' # General app config (currently minimal use)
PROJECT_CONFIG_FILE_TEMPLATE = "{project_name}_config.yaml" # Project-specific config

# --- Configuration Management ---
def load_config():
    """Loads the main application configuration from config.yaml."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)
        if config is None:
            return {}
        return config
    except FileNotFoundError:
        # If the global config file doesn't exist, return empty dict, not an error
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error loading global configuration ({CONFIG_FILE}): {e}")
        return {}

def save_config(config_data):
    """Saves the main application configuration to config.yaml."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    except IOError as e:
        st.error(f"Error saving global configuration ({CONFIG_FILE}): {e}")

def load_project_config(project_base_path, project_name):
    """
    Loads project-specific configuration.
    project_base_path: The directory where the project's folder itself resides.
                       e.g., if project is at 'data/my_project/', then project_base_path is 'data/my_project/'
    """
    config_filename = PROJECT_CONFIG_FILE_TEMPLATE.format(project_name=project_name)
    config_path = os.path.join(project_base_path, config_filename)
    try:
        with open(config_path, 'r') as f:
            project_config = yaml.safe_load(f)
        return project_config if project_config else {}
    except FileNotFoundError:
        # This is expected if a project config hasn't been created yet
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error loading project configuration from '{config_path}': {e}")
        return {}
    except Exception as e:
        st.error(f"An unexpected error occurred while loading project config '{config_path}': {e}")
        return {}


def save_project_config(project_folder_path, project_name, project_config_data):
    """
    Saves project-specific configuration directly into the project's folder.
    project_folder_path: The actual path to the project's dedicated folder (e.g., 'data/my_project_data')
    """
    if not project_name:
        st.error("Project name cannot be empty when saving project configuration.")
        return False
    if not os.path.exists(project_folder_path):
        try:
            os.makedirs(project_folder_path, exist_ok=True)
        except OSError as e:
            st.error(f"Could not create project directory '{project_folder_path}': {e}")
            return False

    config_filename = PROJECT_CONFIG_FILE_TEMPLATE.format(project_name=project_name)
    config_path = os.path.join(project_folder_path, config_filename)

    try:
        with open(config_path, 'w') as f:
            yaml.dump(project_config_data, f, default_flow_style=False)
        # logger.info(f"Project configuration saved to {config_path}") # Use logger if available
        return True
    except IOError as e:
        st.error(f"Error saving project configuration to '{config_path}': {e}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while saving project config to '{config_path}': {e}")
        return False

# --- Logging ---
def setup_logger(name):
    """Sets up a basic logger."""
    logger = logging.getLogger(name)
    # Prevent adding multiple handlers if logger is already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler() # Outputs to console/Streamlit log
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Initialize a logger for this module (optional, but good practice)
# module_logger = setup_logger(__name__)

# --- Error Handling ---
class AppError(Exception):
    """Custom exception class for application-specific errors."""
    pass

def handle_exception(e, context="", logger_instance=None):
    """Handles and logs exceptions, displays error in Streamlit."""
    error_message = f"An error occurred: {context} - {str(e)}"
    if logger_instance:
        logger_instance.error(error_message)
    else:
        # Fallback if no specific logger passed
        fallback_logger = setup_logger("general_exception_handler")
        fallback_logger.error(error_message)
    st.error(error_message)

# --- Other Utilities ---
def generate_project_id(project_name):
    """
    Generates a simple, filesystem-friendly project ID from the project name.
    Example: "My Awesome Project!" -> "my_awesome_project"
    """
    if not project_name:
        return ""
    # Remove special characters, replace spaces with underscores, and lowercase
    # Allow alphanumeric and underscores
    project_id = "".join(c if c.isalnum() else "_" for c in project_name.lower())
    # Replace multiple underscores with a single one
    project_id = "_".join(filter(None, project_id.split("_")))
    return project_id