# modules/auth.py
import streamlit as st
from . import ui_helpers
from . import utils 
import os
import yaml 
import time
# import msal # Removed as OneDrive functionality is removed

logger = utils.setup_logger(__name__) 

# Cloud Authentication Constants and Functions have been removed.

def setup_project_storage(project_name_friendly, storage_type="Local", user_defined_project_dir=None):
    """
    Sets up project storage. All projects are now 'Local'.
    The config file and all data files will be in user_defined_project_dir.
    
    project_name_friendly: The user-chosen, potentially non-filesystem-safe name.
    storage_type: Expected to be "Local". This parameter is kept for structural consistency 
                  but its value is effectively ignored and treated as "Local".
    user_defined_project_dir: The absolute path to the directory where the config file
                              and all project data will be stored.
    Returns True on success, False on failure.
    Updates st.session_state.project_path to user_defined_project_dir.
    """
    if storage_type != "Local":
        logger.info(f"setup_project_storage called with storage_type='{storage_type}'. Defaulting to 'Local' behavior.")
        # storage_type = "Local" # No longer strictly needed to reassign as logic proceeds as if local

    if not project_name_friendly:
        ui_helpers.show_error_message("Project Name cannot be empty.")
        return False
    if not user_defined_project_dir or not os.path.isabs(user_defined_project_dir):
        ui_helpers.show_error_message("A valid absolute path for the Project Directory is required.")
        return False

    project_config_data = st.session_state.get("project_config", {}).copy() 
    project_name_for_file = utils.generate_project_id(project_name_friendly) 

    project_config_data['project_name'] = project_name_friendly 
    project_config_data['storage_type'] = "Local" # Explicitly set to Local
    project_config_data['project_name_for_filename'] = project_name_for_file 
    project_config_data['project_config_file_directory'] = user_defined_project_dir 

    try:
        if not os.path.exists(user_defined_project_dir):
            os.makedirs(user_defined_project_dir, exist_ok=True)
            logger.info(f"Created project directory: {user_defined_project_dir}")
    except OSError as e:
        ui_helpers.show_error_message(f"Could not create project directory '{user_defined_project_dir}': {e}")
        return False

    st.session_state.project_path = user_defined_project_dir 

    if utils.save_project_config(user_defined_project_dir, project_name_for_file, project_config_data):
        st.session_state.project_config = project_config_data 
        msg = f"Project '{project_name_friendly}' (Local) configured. Config and data will be in '{user_defined_project_dir}'."
        logger.info(msg)
        return True
    else:
        logger.error(f"Failed to save project config for '{project_name_friendly}' in '{user_defined_project_dir}'.")
        return False

def store_api_keys(reddit_keys, ai_keys, ai_provider):
    if 'project_config' not in st.session_state or not st.session_state.project_config:
        ui_helpers.show_error_message("Project not active. Cannot save API keys.")
        return False
    
    project_config_to_update = st.session_state.project_config.copy()
    
    project_config_to_update['reddit_api'] = reddit_keys
    project_config_to_update['ai_provider'] = ai_provider
    if ai_provider == "OpenAI":
        project_config_to_update.pop('gemini_api', None)
    elif ai_provider == "Gemini":
        project_config_to_update.pop('openai_api', None)
    project_config_to_update[f'{ai_provider.lower()}_api'] = ai_keys
    
    config_dir = st.session_state.get('project_path') 
    project_name_user_given = project_config_to_update.get('project_name')

    if not config_dir: 
        ui_helpers.show_error_message("Project path (config directory) is not set. Cannot save API keys.")
        return False
    if not project_name_user_given:
        ui_helpers.show_error_message("Project name is missing from current configuration. Cannot save API keys.")
        return False

    if utils.save_project_config(config_dir, project_name_user_given, project_config_to_update):
        st.session_state.project_config = project_config_to_update 
        return True
    else:
        return False

def validate_api_keys(keys, service_name):
    if service_name == "Reddit":
        if keys.get("client_id") and keys.get("client_secret") and keys.get("user_agent"): return True
        else: ui_helpers.show_error_message(f"{service_name} API keys are incomplete. All fields (Client ID, Client Secret, User Agent) are required."); return False
    elif service_name in ["OpenAI", "Gemini"]:
        if keys.get("api_key"): return True
        else: ui_helpers.show_error_message(f"{service_name} API key is missing."); return False
    logger.warning(f"validate_api_keys called with unknown service_name: {service_name}")
    return False