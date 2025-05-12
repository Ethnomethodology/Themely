# app.py
import streamlit as st
import pandas as pd # Keep for any helpers that might remain or for type hints
import os
import yaml
from datetime import datetime

# Import modules - utils and auth are likely needed for project setup on this main page
from modules import auth, utils, ui_helpers 

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    layout="wide",
    page_title="Themely - Qualitative Analysis",
    page_icon="🔬" 
)

logger = utils.setup_logger("app_main_nav")

# --- Initialize Global Session States (used across pages) ---
# Project setup related
if 'project_action_choice' not in st.session_state: st.session_state.project_action_choice = "Create New Project"
if 'current_project_name' not in st.session_state: st.session_state.current_project_name = None
if 'project_config' not in st.session_state: st.session_state.project_config = {}
if 'project_path' not in st.session_state: st.session_state.project_path = None # User-defined dir for config & all local data

# UI input field states (for project setup on this home page)
if 'ui_project_directory_input' not in st.session_state: st.session_state.ui_project_directory_input = os.path.expanduser("~")
if 'ui_project_name_input' not in st.session_state: st.session_state.ui_project_name_input = ""
# if 'ui_storage_type_select_create' not in st.session_state: st.session_state.ui_storage_type_select_create = "Local" # Removed
if 'ui_open_config_file_path_input' not in st.session_state: st.session_state.ui_open_config_file_path_input = ""

# Data-related states that need to be reset when projects change.
# These will be primarily managed by their respective pages but initialized/reset here.
def initialize_project_data_states():
    st.session_state.selected_download_files_info = {}
    st.session_state.combined_data_for_view_creation = pd.DataFrame()
    st.session_state.search_term_combined_data_tab2 = "" 
    st.session_state.redact_confirm_data_management = False 

    st.session_state.selected_project_views_info_tab3 = {}
    st.session_state.data_for_coding_tab3 = pd.DataFrame()
    st.session_state.col_for_coding_tab3 = "text" 
    st.session_state.id_col_for_coding_tab3 = "unique_app_id" 
    st.session_state.search_term_coding_data_tab3 = ""
    st.session_state.currently_selected_view_filepaths_for_saving_tab3 = []
    st.session_state.prompt_save_combined_as_new_view_tab3 = False
    st.session_state.ai_batch_size_tab3 = 10
    st.session_state.ai_batch_prompt_template_p02 = """Perform qualitative thematic analysis on the provided JSON array of data.
Each item in the input array has a "{id_column_name}" and a text field named "{text_column_name}".
For each item, use the text in its "{text_column_name}" field to generate up to 5 concise, meaningful thematic codes.
Return **only** a valid JSON array where each element is an object containing two fields:
1.  "{id_column_name}": the original unique identifier from the input item.
2.  "Codes": a single string of comma-separated thematic codes generated from the text. If no codes are applicable, return an empty string for "Codes".
Example of a single element in the **output** JSON array you should return:
{  "{id_column_name}": "some_original_id_123",  "Codes": "theme alpha, topic beta, user concern gamma"}
Do not include any introductory text, explanations, markdown formatting, or any characters outside the main JSON array structure.
The input data batch you need to process is below:
{json_data_batch}"""
    logger.info("Project-specific data session states initialized/reset by main app.")

# --- Content for the Home Page (Project Setup) ---
def project_setup_page_content():
    st.title("Themely: Qualitative Thematic Analysis")
    st.header("Project Setup & Configuration")

    project_action_key = "project_action_radio_home_nav"
    project_action_index = 0 if st.session_state.get('project_action_choice', "Create New Project") == "Create New Project" else 1
    project_action = st.radio(
        "Choose an action:", ("Create New Project", "Open Existing Project"),
        index=project_action_index, key=project_action_key, horizontal=True,
        on_change=lambda: setattr(st.session_state, 'project_action_choice', st.session_state[project_action_key])
    )
    st.session_state.project_action_choice = st.session_state[project_action_key]

    if st.session_state.project_action_choice == "Create New Project":
        st.subheader("Create New Project")
        st.markdown("Specify the **absolute local directory** for the new project's configuration file and all associated data, and the **project name**.")
        col_dir_cr, col_sep_cr_ui, col_name_cr = st.columns([0.7, 0.05, 0.25])
        with col_dir_cr:
            dir_input_key_cr = "ui_create_dir_input_key_home_nav"
            ui_dir_val_cr = st.text_input(
                "New Project Directory (Absolute Path):", value=st.session_state.ui_project_directory_input, 
                key=dir_input_key_cr, on_change=lambda: setattr(st.session_state, 'ui_project_directory_input', st.session_state[dir_input_key_cr])
            )
            st.session_state.ui_project_directory_input = ui_dir_val_cr
        with col_sep_cr_ui: st.markdown(f"<div style='text-align: center; margin-top: 28px; font-size: 1.2em;'>/</div>", unsafe_allow_html=True) # Placeholder for project name if it were part of path
        with col_name_cr:
            name_input_key_cr = "ui_create_name_input_key_home_nav"
            ui_name_val_cr = st.text_input(
                "New Project Name:", value=st.session_state.ui_project_name_input, 
                key=name_input_key_cr, on_change=lambda: setattr(st.session_state, 'ui_project_name_input', st.session_state[name_input_key_cr])
            )
            st.session_state.ui_project_name_input = ui_name_val_cr

        if st.session_state.ui_project_directory_input and st.session_state.ui_project_name_input:
            sanitized_name_cr_preview = utils.generate_project_id(st.session_state.ui_project_name_input)
            config_filename_cr_preview = utils.PROJECT_CONFIG_FILE_TEMPLATE.format(project_name_id=sanitized_name_cr_preview)
            full_config_path_cr_preview = os.path.join(st.session_state.ui_project_directory_input, config_filename_cr_preview)
            st.caption(f"The project configuration file (`{config_filename_cr_preview}`) and all project data (downloads, views) will be stored in: `{os.path.normpath(st.session_state.ui_project_directory_input)}`")
        
        # Storage type selection removed - defaults to "Local" behavior
        # storage_options_create = ["Local", "Google Drive", "Dropbox", "OneDrive"] 
        # storage_select_key_cr_form = "ui_storage_type_select_create_key_form_home_nav"
        # try: current_storage_idx_cr_form = storage_options_create.index(st.session_state.ui_storage_type_select_create)
        # except ValueError: current_storage_idx_cr_form = 0
        # selected_storage_cr_form = st.selectbox(
        #     "Select Storage Type for New Project:", storage_options_create, index=current_storage_idx_cr_form,
        #     key=storage_select_key_cr_form,
        #     on_change=lambda: setattr(st.session_state, 'ui_storage_type_select_create', st.session_state[storage_select_key_cr_form])
        # )
        # st.session_state.ui_storage_type_select_create = selected_storage_cr_form
        # if selected_storage_cr_form != "Local":
        #     st.info(f"For **{selected_storage_cr_form}**: Data will be on the cloud. Config file in the directory specified above.")

        if st.button("Create New Project", key="create_project_button_action_home_nav"):
            dir_to_create_in_val_form = st.session_state.ui_project_directory_input
            name_to_create_val_form = st.session_state.ui_project_name_input
            storage_for_create_val_form = "Local" # Hardcoded to Local
            if not dir_to_create_in_val_form or not name_to_create_val_form:
                ui_helpers.show_error_message("New Project Directory and New Project Name are required.")
            elif not os.path.isabs(dir_to_create_in_val_form):
                ui_helpers.show_error_message("New Project Directory must be an absolute path.")
            else:
                # The auth.setup_project_storage will now assume 'Local' storage type implicitly.
                # The `user_defined_config_dir` will become the primary project path for everything.
                if auth.setup_project_storage(name_to_create_val_form, storage_type=storage_for_create_val_form, user_defined_project_dir=dir_to_create_in_val_form):
                    st.session_state.current_project_name = name_to_create_val_form
                    st.session_state.project_path = dir_to_create_in_val_form # This IS the project directory
                    
                    # Load config to confirm and ensure all necessary keys are set
                    st.session_state.project_config = utils.load_project_config(dir_to_create_in_val_form, name_to_create_val_form)
                    if not st.session_state.project_config: # Should not happen if setup_project_storage succeeded
                        ui_helpers.show_error_message("Critical error: Project config could not be loaded after creation.")
                        st.session_state.current_project_name = None # Reset state
                        st.session_state.project_path = None
                    else:
                        # Ensure essential keys are in the config after loading
                        st.session_state.project_config['project_name'] = name_to_create_val_form
                        st.session_state.project_config['project_config_file_directory'] = dir_to_create_in_val_form # Redundant but for consistency
                        st.session_state.project_config['storage_type'] = "Local" # Explicitly set
                        utils.save_project_config(dir_to_create_in_val_form, name_to_create_val_form, st.session_state.project_config)

                        initialize_project_data_states() 
                        ui_helpers.show_success_message(f"Project '{name_to_create_val_form}' created successfully in '{dir_to_create_in_val_form}'!")
                        st.rerun()
                else: ui_helpers.show_error_message("Failed to create project. Check directory permissions or logs.")

    elif st.session_state.project_action_choice == "Open Existing Project":
        st.subheader("Open Existing Project")
        open_config_path_key = "ui_open_config_file_path_input_key_home_nav"
        config_file_path_open_val = st.text_input(
            "Absolute Path to Project Configuration File (`..._config.yaml`):",
            value=st.session_state.ui_open_config_file_path_input, key=open_config_path_key,
            on_change=lambda: setattr(st.session_state, 'ui_open_config_file_path_input', st.session_state[open_config_path_key])
        )
        st.session_state.ui_open_config_file_path_input = config_file_path_open_val
        if st.button("Open Project", key="open_project_button_action_home_nav"):
            full_path_to_config_open = st.session_state.ui_open_config_file_path_input
            if not full_path_to_config_open or not os.path.isabs(full_path_to_config_open) or not os.path.isfile(full_path_to_config_open):
                ui_helpers.show_error_message(f"Invalid path or file not found: {full_path_to_config_open}")
            else:
                try:
                    config_dir_to_open = os.path.dirname(full_path_to_config_open)
                    # Attempt to parse project name from filename first (more robust if internal 'project_name' is missing)
                    base_filename = os.path.basename(full_path_to_config_open)
                    project_name_from_filename = None
                    if base_filename.endswith("_config.yaml"):
                        project_name_id_candidate = base_filename[:-len("_config.yaml")]
                        # This doesn't give the user-friendly name, but utils.load_project_config can handle it
                        # by finding the 'project_name' inside the YAML.
                        # For loading, we primarily need the directory and a way to identify the config file.
                        # The internal 'project_name' field is the source of truth for display name.
                    
                    # Load the config using utils.load_project_config which handles finding the actual project name
                    # We need a "user given name" to pass to load_project_config. We can try to infer it
                    # or rely on the content of the YAML.
                    # Let's assume the config file itself contains the 'project_name'.
                    with open(full_path_to_config_open, 'r') as f_direct_open: temp_cfg_open = yaml.safe_load(f_direct_open)
                    
                    project_name_from_cfg_content = temp_cfg_open.get('project_name')

                    if not project_name_from_cfg_content:
                        ui_helpers.show_error_message("Project name not found within the config file. Cannot open.")
                    else:
                        final_loaded_cfg = utils.load_project_config(config_dir_to_open, project_name_from_cfg_content)
                        if not final_loaded_cfg: 
                            ui_helpers.show_error_message(f"Error processing config for '{project_name_from_cfg_content}'. File may be corrupted or structure incorrect.")
                        else:
                            st.session_state.project_config = final_loaded_cfg
                            st.session_state.current_project_name = final_loaded_cfg.get('project_name')
                            # For "Local" projects, project_path is the directory containing the config and data
                            st.session_state.project_path = config_dir_to_open 
                            
                            # Ensure essential keys are consistent
                            st.session_state.project_config['project_config_file_directory'] = config_dir_to_open
                            st.session_state.project_config['storage_type'] = final_loaded_cfg.get('storage_type', "Local") # Read existing or default to Local
                            
                            st.session_state.ui_project_name_input = st.session_state.current_project_name
                            st.session_state.ui_project_directory_input = config_dir_to_open
                            # st.session_state.ui_storage_type_select_create = final_loaded_cfg.get('storage_type', "Local") # UI element removed

                            initialize_project_data_states()
                            ui_helpers.show_success_message(f"Project '{st.session_state.current_project_name}' opened successfully from '{config_dir_to_open}'!")
                            st.rerun()
                except yaml.YAMLError:
                     ui_helpers.show_error_message(f"Error parsing the YAML structure of the config file: {full_path_to_config_open}")
                except Exception as e_open_project: 
                    ui_helpers.show_error_message(f"An error occurred while opening project: {e_open_project}")
                    logger.error(f"Error opening project: {e_open_project}", exc_info=True)
    
    st.markdown("---")
    if st.session_state.current_project_name and st.session_state.project_path:
        active_project_config_dir_info_disp = st.session_state.project_path # This is now the main project directory
        # active_storage_type_info_disp = st.session_state.project_config.get('storage_type', 'Local') # Simplified, always local-like
        st.subheader("Current Active Project")
        st.success(f"**{st.session_state.current_project_name}**")
        # st.caption(f"Storage Type: {active_storage_type_info_disp}") # Not as relevant to display if always local
        st.caption(f"Project Directory (Config & Data): `{os.path.normpath(active_project_config_dir_info_disp)}`")
        
        # Removed OneDrive re-authentication button as cloud is no longer an explicit option
        # if active_storage_type_info_disp == "OneDrive": 
        #     if st.button("Re-authenticate OneDrive", key="reauth_onedrive_active_project_key_home_nav"): 
        #         st.session_state.force_onedrive_reauth = True
        #         # The auth.setup_project_storage call might need adjustment if it strictly expects a user_defined_config_dir
        #         # For re-auth, it would use the existing project_path.
        #         if auth.setup_project_storage(st.session_state.current_project_name, "OneDrive", user_defined_project_dir=st.session_state.project_path):
        #              st.session_state.project_config = utils.load_project_config(st.session_state.project_path, st.session_state.current_project_name)
        #              ui_helpers.show_success_message("OneDrive re-authentication process completed.")
        #         else: ui_helpers.show_error_message("OneDrive re-authentication process failed or was cancelled.")
        #         if 'force_onedrive_reauth' in st.session_state: del st.session_state['force_onedrive_reauth']
        #         st.rerun()
    else:
        st.info("No project is currently active. Create a new project or open an existing one above.")

    st.divider()
    st.header("API Key Management")
    if st.session_state.current_project_name and st.session_state.project_path:
        # API keys are always stored in the _config.yaml within the project_path
        api_config_for_display = utils.load_project_config(st.session_state.project_path, st.session_state.current_project_name)
        if not api_config_for_display: api_config_for_display = st.session_state.project_config # Fallback to session state if load failed

        with st.expander("Reddit API Credentials", expanded=False):
            reddit_cfg_api_form = api_config_for_display.get('reddit_api', {})
            client_id_api_form = st.text_input("Reddit Client ID", type="password", value=reddit_cfg_api_form.get('client_id', ''), key="api_reddit_client_id_form_home_nav")
            client_secret_api_form = st.text_input("Reddit Client Secret", type="password", value=reddit_cfg_api_form.get('client_secret', ''), key="api_reddit_client_secret_form_home_nav")
            user_agent_default_form = f'Python:ThemelyApp_{utils.generate_project_id(st.session_state.current_project_name)}:v0.1 by /u/YourUsername'
            user_agent_api_form = st.text_input("Reddit User Agent", value=reddit_cfg_api_form.get('user_agent', user_agent_default_form), key="api_reddit_user_agent_form_home_nav")

        with st.expander("Generative AI API Credentials", expanded=False):
            ai_provider_default_api_form = api_config_for_display.get('ai_provider', 'OpenAI')
            try: ai_provider_idx_api_keys_form = ["OpenAI", "Gemini"].index(ai_provider_default_api_form)
            except ValueError: ai_provider_idx_api_keys_form = 0 # Default to OpenAI if current value is somehow invalid
            
            current_ai_provider_val = api_config_for_display.get('ai_provider', 'OpenAI')
            # Ensure the index is valid for the selectbox
            try:
                current_ai_provider_idx = ["OpenAI", "Gemini"].index(current_ai_provider_val)
            except ValueError:
                current_ai_provider_idx = 0 # Default to OpenAI if stored value is not in list
                
            ai_provider_select_api_form = st.selectbox(
                "Select AI Provider", 
                ["OpenAI", "Gemini"], 
                index=current_ai_provider_idx, 
                key="api_ai_provider_select_form_home_nav"
            )
            
            ai_key_val_api_form = api_config_for_display.get(f'{ai_provider_select_api_form.lower()}_api', {}).get('api_key', '')
            ai_key_input_api_form = st.text_input(
                f"{ai_provider_select_api_form} API Key", 
                type="password", 
                value=ai_key_val_api_form, 
                key=f"api_ai_key_input_form_home_nav_{ai_provider_select_api_form}" # Key changes with provider to reset value
            )

        if st.button("Save API Keys", key="api_save_keys_button_form_home_nav"):
            r_keys_form = {"client_id": client_id_api_form, "client_secret": client_secret_api_form, "user_agent": user_agent_api_form}
            gai_keys_form = {"api_key": ai_key_input_api_form}
            
            # Perform basic validation on input fields before attempting to store
            valid_reddit = all(r_keys_form.values()) # Simple check if all fields have some value
            valid_gai = bool(gai_keys_form["api_key"])

            if not valid_reddit:
                ui_helpers.show_warning_message("Reddit API fields cannot be empty. Keys not saved.")
            elif not valid_gai:
                 ui_helpers.show_warning_message(f"{ai_provider_select_api_form} API Key cannot be empty. Keys not saved.")
            elif auth.validate_api_keys(r_keys_form, "Reddit") and auth.validate_api_keys(gai_keys_form, ai_provider_select_api_form):
                # auth.store_api_keys uses project_path and project_name from session_state
                if auth.store_api_keys(r_keys_form, gai_keys_form, ai_provider_select_api_form): 
                    # Reload config into session state after saving
                    st.session_state.project_config = utils.load_project_config(st.session_state.project_path, st.session_state.current_project_name)
                    ui_helpers.show_success_message("API keys saved successfully to project configuration.")
                    st.rerun() # Rerun to reflect changes in input fields if they were masked
                # else: ui_helpers.show_error_message("Failed to save API keys due to an internal error.") # Covered by store_api_keys
            # else: ui_helpers.show_warning_message("API key validation failed (internal check). Keys not saved.") # Covered by validate_api_keys
    else:
        st.info("Create or open a project to manage its API keys.")

# --- Sidebar Setup ---
st.sidebar.title("Themely") 
st.sidebar.caption("Qualitative Analysis Assistant")
st.sidebar.divider()

# --- Navigation Definition ---
page_definitions = [
    st.Page(project_setup_page_content, title="Project Setup", icon="🏠", default=True),
    st.Page("pages/data_management_page.py", title="Data Management", icon="💾"),
    st.Page("pages/ai_coding_page.py", title="Coding", icon="🤖"),
    st.Page("pages/analysis_page.py", title="Analysis", icon="📊"),
]
pg = st.navigation(page_definitions)

# --- Common Sidebar Elements (after navigation) ---
st.sidebar.header("Active Project")
if st.session_state.get('current_project_name') and st.session_state.get('project_path'):
    st.sidebar.success(f"{st.session_state.current_project_name}")
    project_sidebar_path_display = st.session_state.project_path
    # active_storage_sidebar_val = st.session_state.get('project_config', {}).get('storage_type', 'Local') # Always Local now
    # st.sidebar.caption(f"Type: {active_storage_sidebar_val}")
    st.sidebar.caption(f"Project Dir: `{os.path.normpath(project_sidebar_path_display)}`")
else:
    st.sidebar.warning("No active project.")
st.sidebar.divider()
st.sidebar.header("About Themely")
st.sidebar.info(
    """
    This tool assists with qualitative thematic analysis of Reddit data.
    Use the navigation menu to move between stages.
    All project data is stored locally in the specified project directory.
    """
)

# --- Execute the selected page ---
pg.run()

# Recovery logic - ensure project_path and project_config_file_directory are aligned
if st.session_state.current_project_name:
    if not st.session_state.project_path and st.session_state.project_config.get('project_config_file_directory'):
        # If project_path is missing but directory is in config, try to restore project_path
        logger.warning("project_path missing, attempting recovery from project_config['project_config_file_directory']")
        st.session_state.project_path = st.session_state.project_config['project_config_file_directory']
        # initialize_project_data_states() # Might not be needed if only project_path was lost
        # st.rerun()

    elif st.session_state.project_path and \
         st.session_state.project_config.get('project_config_file_directory') != st.session_state.project_path:
        # If they are different, project_path (derived from user input) should be the source of truth for "Local" projects.
        # However, load_project_config already ensures project_config_file_directory is updated.
        # This case might indicate an inconsistent state that load_project_config should resolve.
        logger.warning(f"project_path ('{st.session_state.project_path}') and config's dir ('{st.session_state.project_config.get('project_config_file_directory')}') mismatch. Attempting to reload config.")
        loaded_cfg_recovery_main = utils.load_project_config(st.session_state.project_path, st.session_state.current_project_name)
        if loaded_cfg_recovery_main:
            st.session_state.project_config = loaded_cfg_recovery_main
            # project_path should already be correct from user input or opening logic
            st.session_state.project_config['project_config_file_directory'] = st.session_state.project_path
            st.session_state.project_config['storage_type'] = "Local" # Enforce local
            utils.save_project_config(st.session_state.project_path, st.session_state.current_project_name, st.session_state.project_config)
            # initialize_project_data_states() # Consider if a full reset is needed
            # st.rerun() # May cause loop if not careful
        else:
            logger.error(f"Recovery failed for {st.session_state.current_project_name} due to config load error from project_path.")
            # Potentially clear project state if recovery fails badly
            # st.session_state.current_project_name = None
            # st.session_state.project_path = None
            # st.session_state.project_config = {}
            # ui_helpers.show_error_message("Project state inconsistent and recovery failed. Please try reopening the project.")
            # st.rerun()