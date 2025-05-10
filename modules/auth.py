# modules/auth.py
import streamlit as st
from . import ui_helpers
from . import utils
import os
import json
import time
import msal

logger = utils.setup_logger(__name__)

ONEDRIVE_CLIENT_ID = "YOUR_ACTUAL_ONEDRIVE_APP_CLIENT_ID" # Replace this
ONEDRIVE_AUTHORITY = "https://login.microsoftonline.com/common"
ONEDRIVE_SCOPES = ["Files.ReadWrite.AppFolder", "User.Read", "offline_access"]

def authenticate_google_drive():
    st.info("Google Drive authentication flow would be initiated here (conceptual).")
    return {"access_token": "dummy_google_token", "provider": "Google Drive"}

def authenticate_dropbox():
    st.info("Dropbox authentication flow would be initiated here (conceptual).")
    return {"access_token": "dummy_dropbox_token", "provider": "Dropbox"}

def load_onedrive_token_from_project_config():
    if 'project_config' in st.session_state and st.session_state.project_config:
        return st.session_state.project_config.get('onedrive_token_cache')
    return None

def save_onedrive_token_to_project_config(token_cache_data):
    if 'project_config' not in st.session_state or not st.session_state.project_config:
        ui_helpers.show_error_message("Project not set up. Cannot save OneDrive token.")
        return False
    st.session_state.project_config['onedrive_token_cache'] = token_cache_data
    project_actual_config_folder_path = st.session_state.project_config.get('project_config_actual_path') # where the yaml is stored
    project_name = st.session_state.project_config.get('project_name')
    if project_actual_config_folder_path and project_name:
        return utils.save_project_config(project_actual_config_folder_path, project_name, st.session_state.project_config)
    else:
        ui_helpers.show_error_message("Project config path or name not found for saving token.")
        return False

def authenticate_onedrive(force_reauth=False):
    if not ONEDRIVE_CLIENT_ID or ONEDRIVE_CLIENT_ID == "YOUR_ACTUAL_ONEDRIVE_APP_CLIENT_ID":
        ui_helpers.show_error_message("OneDrive Client ID is not configured in modules/auth.py. Please update it.")
        st.stop()
    app = msal.PublicClientApplication(ONEDRIVE_CLIENT_ID, authority=ONEDRIVE_AUTHORITY)
    if not force_reauth:
        token_cache_data = load_onedrive_token_from_project_config()
        if token_cache_data and isinstance(token_cache_data, dict) and token_cache_data.get("accounts"):
            try:
                # Simplified account finding; MSAL's cache is more complex.
                # This assumes the first account is desired if multiple were ever cached (unlikely for device flow per app instance)
                account_to_use = app.get_accounts(username=token_cache_data["accounts"][0].get("username"))
                if account_to_use:
                    result = app.acquire_token_silent(ONEDRIVE_SCOPES, account=account_to_use[0])
                    if result and "access_token" in result:
                        ui_helpers.show_success_message("Re-authenticated with OneDrive using cached token.")
                        return {"access_token": result["access_token"], "token_cache": token_cache_data, "provider": "OneDrive"}
            except Exception as e:
                logger.warning(f"Error during OneDrive silent token acquisition: {e}. Proceeding to interactive auth.")

    flow = app.initiate_device_flow(scopes=ONEDRIVE_SCOPES)
    if "user_code" not in flow:
        error_desc = flow.get("error_description", "Unknown error during device flow initiation.")
        ui_helpers.show_error_message(f"Failed to create device flow for OneDrive: {error_desc}")
        logger.error(f"OneDrive device flow initiation error: {flow.get('error', '')} - {error_desc}")
        return None
    st.info(f"To authenticate OneDrive:\n1. Go to: {flow['verification_uri']}\n2. Enter code: {flow['user_code']}")
    st.markdown(f"<a href='{flow['verification_uri']}' target='_blank'>Open auth page</a> (code: {flow['user_code']})", unsafe_allow_html=True)
    result = None
    with st.spinner("Waiting for OneDrive authentication in browser..."):
        try:
            result = app.acquire_token_by_device_flow(flow, timeout=180)
        except Exception as e:
            ui_helpers.show_error_message(f"Error during OneDrive device flow: {e}")
            logger.error(f"OneDrive acquire_token_by_device_flow error: {e}")
            return None
    if result and "access_token" in result:
        ui_helpers.show_success_message("Successfully authenticated with OneDrive!")
        # Simplified cache for re-auth attempts. MSAL's full cache object is complex.
        simplified_cache_data = {
            "accounts": [{"username": acc.get("username"), "local_account_id": acc.get("local_account_id")} for acc in app.get_accounts()],
            "access_token_info": {k: v for k, v in result.items() if k not in ["access_token", "id_token"]} # Avoid storing tokens directly if possible
        }
        if save_onedrive_token_to_project_config(simplified_cache_data):
             logger.info("OneDrive token information concept saved.")
        return {"access_token": result["access_token"], "token_cache": simplified_cache_data, "provider": "OneDrive"}
    elif result and "error" in result:
        ui_helpers.show_error_message(f"OneDrive Auth Error: {result.get('error_description', result.get('error'))}")
    else:
        ui_helpers.show_error_message("OneDrive authentication failed/cancelled.")
    return None

def setup_project_storage(project_name, storage_type, designated_path):
    """
    Sets up project storage.
    designated_path: For "Local", this is the user-chosen absolute base path.
                     For "Cloud", this is the base path where local *config* files for cloud projects are stored.
    Returns True on success, False on failure.
    Updates st.session_state.project_path to the actual project folder (for local)
    or the folder containing the project's config yaml (for cloud).
    """
    project_config_data = st.session_state.get("project_config", {}).copy() # Work on a copy
    project_id = utils.generate_project_id(project_name)
    project_config_data['project_name'] = project_name
    project_config_data['storage_type'] = storage_type
    project_config_data['project_id'] = project_id

    actual_project_folder_path = None # This will be where data/views are stored for Local, or where config is for Cloud

    if storage_type == "Local":
        if not designated_path or not os.path.isabs(designated_path):
            ui_helpers.show_error_message("Local storage requires a valid absolute base path.")
            return False
        if not os.path.isdir(designated_path):
            try:
                os.makedirs(designated_path, exist_ok=True)
                logger.info(f"Created base directory for local projects: {designated_path}")
            except OSError as e:
                ui_helpers.show_error_message(f"Could not create base directory '{designated_path}': {e}")
                return False
        
        actual_project_folder_path = os.path.join(designated_path, project_id)
        project_config_data['project_path_on_disk'] = actual_project_folder_path # Actual data path for local
        project_config_data['user_local_base_path_for_projects'] = designated_path # Store user's choice

    elif storage_type in ["Google Drive", "Dropbox", "OneDrive"]:
        # For cloud storage, 'designated_path' is where the local .yaml config for this cloud project will be stored.
        # The actual project data is on the cloud.
        if not designated_path: # Should be 'data/project_configs_for_cloud_storage/'
            ui_helpers.show_error_message("Base path for storing cloud project configurations is missing.")
            return False
        
        actual_project_folder_path = os.path.join(designated_path, project_id) # Folder for this specific cloud project's local config
        project_config_data['project_path_on_cloud_conceptual'] = f"{storage_type}_root/{project_id}" # Conceptual data path

    else:
        ui_helpers.show_error_message(f"Unsupported storage type: {storage_type}")
        return False

    # Ensure the directory for the project (local data or local cloud config) exists
    try:
        if not os.path.exists(actual_project_folder_path):
            os.makedirs(actual_project_folder_path, exist_ok=True)
    except OSError as e:
        ui_helpers.show_error_message(f"Failed to create project directory at '{actual_project_folder_path}': {e}")
        return False

    # Store the path where the project's YAML config file itself will be/is located.
    # For local storage, this is actual_project_folder_path.
    # For cloud storage, this is also actual_project_folder_path (which is like data/project_configs_for_cloud_storage/project_id/).
    project_config_data['project_config_actual_path'] = actual_project_folder_path
    st.session_state.project_path = actual_project_folder_path # Critical for other modules to find config/data

    auth_success_for_cloud = True # Assume true for local, check for cloud
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
            # Add similar for other cloud providers if they return specific cache/token info
            # project_config_data[f'{storage_type.lower().replace(" ", "_")}_credentials'] = creds # Generic placeholder
        else:
            auth_success_for_cloud = False
            ui_helpers.show_warning_message(f"Authentication for {storage_type} was not completed or failed. You can proceed to save API keys, but cloud features requiring this auth might not work.")


    # Save the project configuration (project_config_data) into actual_project_folder_path/project_name_config.yaml
    if utils.save_project_config(actual_project_folder_path, project_name, project_config_data):
        st.session_state.project_config = project_config_data # Update session with potentially new paths/tokens
        msg = f"Project '{project_name}' ({storage_type}) configured successfully."
        if storage_type != "Local" and not auth_success_for_cloud:
            msg += " Note: Cloud authentication was not completed."
        ui_helpers.show_success_message(msg)
        return True
    else:
        # Error message handled by save_project_config
        return False


def store_api_keys(reddit_keys, ai_keys, ai_provider):
    if 'project_config' not in st.session_state or not st.session_state.project_config:
        ui_helpers.show_error_message("Project not set up. Please set up a project first.")
        return False
    project_config = st.session_state.project_config
    project_config['reddit_api'] = reddit_keys
    project_config['ai_provider'] = ai_provider
    project_config[f'{ai_provider.lower()}_api'] = ai_keys
    
    # project_config_actual_path is the folder where the _config.yaml is stored
    config_folder_path = project_config.get('project_config_actual_path') 
    project_name = project_config.get('project_name')

    if not config_folder_path or not project_name:
        ui_helpers.show_error_message("Project configuration path or name is missing. Cannot save API keys.")
        return False
    if utils.save_project_config(config_folder_path, project_name, project_config):
        st.session_state.project_config = project_config # Update session state
        ui_helpers.show_success_message("API keys saved successfully in project configuration.")
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