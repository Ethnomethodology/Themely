# modules/auth.py
import streamlit as st
from . import ui_helpers
from . import utils # utils.py now has load_project_config_from_file
import os
import yaml # For loading config if needed, though utils handles it mostly
import time
import msal

# Use the logger from utils to avoid re-declaration issues
logger = utils.setup_logger(__name__) # Use a distinct name for this module's logger

ONEDRIVE_CLIENT_ID = "YOUR_ACTUAL_ONEDRIVE_APP_CLIENT_ID" # Replace this
ONEDRIVE_AUTHORITY = "https://login.microsoftonline.com/common"
ONEDRIVE_SCOPES = ["Files.ReadWrite.AppFolder", "User.Read", "offline_access"]

def authenticate_google_drive():
    st.info("Google Drive authentication flow would be initiated here (conceptual).")
    return {"access_token": "dummy_google_token", "provider": "Google Drive"}

def authenticate_dropbox():
    st.info("Dropbox authentication flow would be initiated here (conceptual).")
    return {"access_token": "dummy_dropbox_token", "provider": "Dropbox"}

def load_onedrive_token_from_project_config(): # This can stay as it reads from loaded session_state.project_config
    if 'project_config' in st.session_state and st.session_state.project_config:
        return st.session_state.project_config.get('onedrive_token_cache')
    return None

def save_onedrive_token_to_project_config(token_cache_data): # This also works with loaded project_config
    if 'project_config' not in st.session_state or not st.session_state.project_config:
        ui_helpers.show_error_message("Project not set up. Cannot save OneDrive token.")
        return False
    st.session_state.project_config['onedrive_token_cache'] = token_cache_data
    
    config_file_dir = st.session_state.project_config.get('project_config_file_directory')
    project_name = st.session_state.project_config.get('project_name') # User-friendly name
    project_name_for_file = utils.generate_project_id(project_name) # Sanitized name for filename

    if config_file_dir and project_name_for_file:
        # utils.save_project_config expects directory and the sanitized name for the filename part
        return utils.save_project_config(config_file_dir, project_name_for_file, st.session_state.project_config)
    else:
        ui_helpers.show_error_message("Project config directory or name not found for saving token.")
        return False

def authenticate_onedrive(force_reauth=False):
    # ... (OneDrive authentication logic remains the same as before) ...
    if not ONEDRIVE_CLIENT_ID or ONEDRIVE_CLIENT_ID == "YOUR_ACTUAL_ONEDRIVE_APP_CLIENT_ID":
        ui_helpers.show_error_message("OneDrive Client ID is not configured in modules/auth.py. Please update it.")
        st.stop()
    app = msal.PublicClientApplication(ONEDRIVE_CLIENT_ID, authority=ONEDRIVE_AUTHORITY)
    if not force_reauth:
        token_cache_data = load_onedrive_token_from_project_config()
        if token_cache_data and isinstance(token_cache_data, dict) and token_cache_data.get("accounts"):
            try:
                accounts_info = token_cache_data["accounts"]
                if accounts_info:
                    username_to_find = accounts_info[0].get("username")
                    account_to_use = app.get_accounts(username=username_to_find) if username_to_find else app.get_accounts()
                    if account_to_use:
                        result = app.acquire_token_silent(ONEDRIVE_SCOPES, account=account_to_use[0])
                        if result and "access_token" in result:
                            return {"access_token": result["access_token"], "token_cache": token_cache_data, "provider": "OneDrive"}
            except Exception as e:
                logger.warning(f"Error during OneDrive silent token acquisition: {e}. Proceeding to interactive auth.")

    flow = app.initiate_device_flow(scopes=ONEDRIVE_SCOPES)
    if "user_code" not in flow:
        error_desc = flow.get("error_description", "Unknown error during device flow initiation.")
        ui_helpers.show_error_message(f"Failed to create device flow for OneDrive: {error_desc}")
        return None
    st.info(f"To authenticate OneDrive:\n1. Go to: {flow['verification_uri']}\n2. Enter code: {flow['user_code']}")
    st.markdown(f"<a href='{flow['verification_uri']}' target='_blank'>Open auth page</a> (code: {flow['user_code']})", unsafe_allow_html=True)
    result = None
    with st.spinner("Waiting for OneDrive authentication in browser..."):
        try: result = app.acquire_token_by_device_flow(flow, timeout=180)
        except Exception as e: ui_helpers.show_error_message(f"Error during OneDrive device flow: {e}"); return None
    
    if result and "access_token" in result:
        ui_helpers.show_success_message("Successfully authenticated with OneDrive!")
        current_accounts = app.get_accounts()
        simplified_cache_data = {
            "accounts": [{"username": acc.get("username"), "local_account_id": acc.get("local_account_id"), "home_account_id": acc.get("home_account_id")} for acc in current_accounts] if current_accounts else [],
            "access_token_info": {k: v for k, v in result.items() if k not in ["access_token", "id_token", "refresh_token"]}
        }
        if save_onedrive_token_to_project_config(simplified_cache_data): # This will use the new save logic
             logger.info("OneDrive token information concept saved to project config.")
        return {"access_token": result["access_token"], "token_cache": simplified_cache_data, "provider": "OneDrive"}
    elif result and "error" in result: ui_helpers.show_error_message(f"OneDrive Auth Error: {result.get('error_description', result.get('error'))}")
    else: ui_helpers.show_error_message("OneDrive authentication failed/cancelled.")
    return None


def setup_project_storage(project_name_friendly, storage_type, user_defined_config_dir):
    """
    Sets up project storage. The config file will be in user_defined_config_dir.
    For "Local" storage, data files will also be in user_defined_config_dir.
    user_defined_config_dir: The absolute path to the directory where the config file
                             (and local data) will be stored.
    project_name_friendly: The user-chosen, potentially non-filesystem-safe name.
    Returns True on success, False on failure.
    Updates st.session_state.project_path to user_defined_config_dir.
    """
    if not project_name_friendly:
        ui_helpers.show_error_message("Project Name cannot be empty.")
        return False
    if not user_defined_config_dir or not os.path.isabs(user_defined_config_dir):
        ui_helpers.show_error_message("A valid absolute path for the Project/Config Directory is required.")
        return False

    project_config_data = st.session_state.get("project_config", {}).copy()
    project_name_for_file = utils.generate_project_id(project_name_friendly) # For config filename

    project_config_data['project_name'] = project_name_friendly # Store user-friendly name
    project_config_data['storage_type'] = storage_type
    project_config_data['project_name_for_filename'] = project_name_for_file # Store sanitized name used for filename
    project_config_data['project_config_file_directory'] = user_defined_config_dir # Dir where config lives

    try:
        if not os.path.exists(user_defined_config_dir):
            os.makedirs(user_defined_config_dir, exist_ok=True)
            logger.info(f"Created project/config directory: {user_defined_config_dir}")
    except OSError as e:
        ui_helpers.show_error_message(f"Could not create directory '{user_defined_config_dir}': {e}")
        return False

    st.session_state.project_path = user_defined_config_dir # CRITICAL update for other modules

    if storage_type != "Local":
        project_config_data['project_path_on_cloud_conceptual'] = f"{storage_type}_root/{project_name_for_file}"

    auth_success_for_cloud = True
    if storage_type in ["Google Drive", "Dropbox", "OneDrive"]:
        creds = None
        if storage_type == "Google Drive": creds = authenticate_google_drive()
        elif storage_type == "Dropbox": creds = authenticate_dropbox()
        elif storage_type == "OneDrive":
            creds = authenticate_onedrive(force_reauth=st.session_state.get("force_onedrive_reauth", False))
            if 'force_onedrive_reauth' in st.session_state: del st.session_state['force_onedrive_reauth']
        
        if creds and creds.get("access_token"):
            if storage_type == "OneDrive" and "token_cache" in creds:
                project_config_data['onedrive_token_cache'] = creds['token_cache']
        else:
            auth_success_for_cloud = False

    # Save the initial project configuration
    # utils.save_project_config expects directory and the sanitized name for the filename part
    if utils.save_project_config(user_defined_config_dir, project_name_for_file, project_config_data):
        st.session_state.project_config = project_config_data
        msg = f"Project '{project_name_friendly}' ({storage_type}) configured. Config file in '{user_defined_config_dir}'."
        if storage_type != "Local" and not auth_success_for_cloud:
            msg += " Note: Cloud authentication may not have completed."
        ui_helpers.show_success_message(msg)
        return True
    else:
        return False

def store_api_keys(reddit_keys, ai_keys, ai_provider):
    if 'project_config' not in st.session_state or not st.session_state.project_config:
        ui_helpers.show_error_message("Project not active. Cannot save API keys.")
        return False
    project_config_to_update = st.session_state.project_config.copy()
    project_config_to_update['reddit_api'] = reddit_keys
    project_config_to_update['ai_provider'] = ai_provider
    project_config_to_update[f'{ai_provider.lower()}_api'] = ai_keys
    
    config_dir = project_config_to_update.get('project_config_file_directory')
    # Use the sanitized name for saving the file, as stored in the config
    project_name_for_file_save = project_config_to_update.get('project_name_for_filename', utils.generate_project_id(project_config_to_update.get('project_name')))


    if not config_dir or not project_name_for_file_save:
        ui_helpers.show_error_message("Project configuration directory or name for filename is missing. Cannot save API keys.")
        return False
    if utils.save_project_config(config_dir, project_name_for_file_save, project_config_to_update):
        st.session_state.project_config = project_config_to_update
        ui_helpers.show_success_message("API keys saved to project configuration.")
        return True
    else:
        return False

def validate_api_keys(keys, service_name):
    if service_name == "Reddit":
        if keys.get("client_id") and keys.get("client_secret") and keys.get("user_agent"): return True
        else: ui_helpers.show_error_message(f"{service_name} API keys are incomplete."); return False
    elif service_name in ["OpenAI", "Gemini"]:
        if keys.get("api_key"): return True
        else: ui_helpers.show_error_message(f"{service_name} API key is missing."); return False
    return False