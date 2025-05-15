# app.py
import streamlit as st
import pandas as pd # Keep for any helpers that might remain or for type hints
import os
import yaml
from datetime import datetime
import shutil

# Import modules - utils and auth are likely needed for project setup on this main page
from modules import auth, utils, ui_helpers
from modules import db
# Initialize database and load past project list
db.init_db()
if 'past_projects' not in st.session_state:
    st.session_state.past_projects = db.load_project_index()
if 'removed_projects' not in st.session_state:
    st.session_state.removed_projects = []
# Filter out any removed names
st.session_state.past_projects = [
    p for p in st.session_state.past_projects
    if p['name'] not in st.session_state.removed_projects
]

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    layout="wide",
    page_title="Themely - Qualitative Analysis"
)

logger = utils.setup_logger("app_main_nav")

# --- Initialize Global Session States (used across pages) ---
# Project setup related
if 'project_action_choice' not in st.session_state:
    st.session_state.project_action_choice = "New Project"
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
    st.title("Themely")
    st.header("Project Setup")
    st.write("Get started by creating a new project or opening an existing one, then enter your Reddit and AI service API keys.")
    ui_helpers.page_sidebar_info([
        "Create or open a project",
        "Enter Reddit API credentials",
        "Configure AI provider"
    ])
    # Begin two-column layout
    left_col, right_col = st.columns([1, 1])

    # LEFT COLUMN: Create/Open controls
    with left_col:
        project_action_key = "project_action_radio_home_nav"
        project_action_index = 0 if st.session_state.get('project_action_choice', "New Project") == "New Project" else 1
        project_action = st.radio(
            "Action:", ("New Project", "Open Project"),
            index=project_action_index, key=project_action_key, horizontal=True,
            on_change=lambda: setattr(st.session_state, 'project_action_choice', st.session_state[project_action_key])
        )
        st.session_state.project_action_choice = st.session_state[project_action_key]

        if st.session_state.project_action_choice == "New Project":
            st.markdown("Set project name and directory.")
            col_dir_cr, col_sep_cr_ui, col_name_cr = st.columns([0.7, 0.05, 0.25])
            with col_dir_cr:
                dir_input_key_cr = "ui_create_dir_input_key_home_nav"
                ui_dir_val_cr = st.text_input(
                    "Directory:", value=st.session_state.ui_project_directory_input, 
                    key=dir_input_key_cr, on_change=lambda: setattr(st.session_state, 'ui_project_directory_input', st.session_state[dir_input_key_cr])
                )
                st.session_state.ui_project_directory_input = ui_dir_val_cr
            with col_sep_cr_ui:
                st.markdown(f"<div style='text-align: center; margin-top: 28px; font-size: 1.2em;'>/</div>", unsafe_allow_html=True)
            with col_name_cr:
                name_input_key_cr = "ui_create_name_input_key_home_nav"
                ui_name_val_cr = st.text_input(
                    "Project name:", value=st.session_state.ui_project_name_input, 
                    key=name_input_key_cr, on_change=lambda: setattr(st.session_state, 'ui_project_name_input', st.session_state[name_input_key_cr])
                )
                st.session_state.ui_project_name_input = ui_name_val_cr

            if st.session_state.ui_project_directory_input and st.session_state.ui_project_name_input:
                sanitized_name_cr_preview = utils.generate_project_id(st.session_state.ui_project_name_input)
                config_filename_cr_preview = utils.PROJECT_CONFIG_FILE_TEMPLATE.format(project_name_id=sanitized_name_cr_preview)
                # The preview path is now where the config would go, after the project folder is created
                full_config_path_cr_preview = os.path.join(
                    os.path.join(st.session_state.ui_project_directory_input, sanitized_name_cr_preview),
                    config_filename_cr_preview
                )

            if st.button("Create", key="create_project_button_action_home_nav"):
                dir_to_create_in_val_form = st.session_state.ui_project_directory_input
                name_to_create_val_form = st.session_state.ui_project_name_input
                storage_for_create_val_form = "Local" # Hardcoded to Local
                if not dir_to_create_in_val_form or not name_to_create_val_form:
                    ui_helpers.show_error_message("New Project Directory and New Project Name are required.")
                elif not os.path.isabs(dir_to_create_in_val_form):
                    ui_helpers.show_error_message("New Project Directory must be an absolute path.")
                else:
                    # Create project directory inside the chosen directory, named after project ID
                    project_id = utils.generate_project_id(name_to_create_val_form)
                    project_folder = os.path.join(dir_to_create_in_val_form, project_id)
                    try:
                        os.makedirs(project_folder, exist_ok=True)
                    except Exception as e:
                        ui_helpers.show_error_message(f"Failed to create project folder: {e}")
                        return
                    # Call setup_project_storage with project_folder as directory
                    if auth.setup_project_storage(
                        name_to_create_val_form,
                        storage_type=storage_for_create_val_form,
                        user_defined_project_dir=project_folder
                    ):
                        st.session_state.current_project_name = name_to_create_val_form
                        st.session_state.project_path = project_folder
                        # Save project index using config path inside project_folder
                        config_filename_cr_preview = utils.PROJECT_CONFIG_FILE_TEMPLATE.format(project_name_id=project_id)
                        full_config_path_cr_preview = os.path.join(project_folder, config_filename_cr_preview)
                        db.save_project_index(name_to_create_val_form, full_config_path_cr_preview)
                        st.session_state.past_projects = [
                            p for p in db.load_project_index()
                            if p['name'] not in st.session_state.removed_projects
                        ]
                        st.session_state.project_config = utils.load_project_config(project_folder, name_to_create_val_form)
                        if not st.session_state.project_config:
                            ui_helpers.show_error_message("Critical error: Project config could not be loaded after creation.")
                            st.session_state.current_project_name = None
                            st.session_state.project_path = None
                        else:
                            st.session_state.project_config['project_name'] = name_to_create_val_form
                            st.session_state.project_config['project_config_file_directory'] = project_folder
                            st.session_state.project_config['storage_type'] = "Local"
                            # Save config YAML inside project_folder
                            utils.save_project_config(project_folder, name_to_create_val_form, st.session_state.project_config)
                            initialize_project_data_states()
                            ui_helpers.show_success_message(f"Project '{name_to_create_val_form}' created successfully in '{project_folder}'!")
                            st.rerun()
                    else:
                        ui_helpers.show_error_message("Failed to create project. Check directory permissions or logs.")

        elif st.session_state.project_action_choice == "Open Project":
            open_config_path_key = "ui_open_config_file_path_input_key_home_nav"
            config_file_path_open_val = st.text_input(
                "Project config file:",
                value=st.session_state.ui_open_config_file_path_input, key=open_config_path_key,
                on_change=lambda: setattr(st.session_state, 'ui_open_config_file_path_input', st.session_state[open_config_path_key])
            )
            st.session_state.ui_open_config_file_path_input = config_file_path_open_val
            if st.button("Open", key="open_project_button_action_home_nav"):
                full_path_to_config_open = st.session_state.ui_open_config_file_path_input
                if not full_path_to_config_open or not os.path.isabs(full_path_to_config_open) or not os.path.isfile(full_path_to_config_open):
                    ui_helpers.show_error_message(f"Invalid path or file not found: {full_path_to_config_open}")
                else:
                    try:
                        config_dir_to_open = os.path.dirname(full_path_to_config_open)
                        base_filename = os.path.basename(full_path_to_config_open)
                        project_name_from_filename = None
                        if base_filename.endswith("_config.yaml"):
                            project_name_id_candidate = base_filename[:-len("_config.yaml")]
                        with open(full_path_to_config_open, 'r') as f_direct_open:
                            temp_cfg_open = yaml.safe_load(f_direct_open)
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
                                st.session_state.project_path = config_dir_to_open
                                db.save_project_index(st.session_state.current_project_name, full_path_to_config_open)
                                st.session_state.past_projects = [
                                    p for p in db.load_project_index()
                                    if p['name'] not in st.session_state.removed_projects
                                ]
                                st.session_state.project_config['project_config_file_directory'] = config_dir_to_open
                                st.session_state.project_config['storage_type'] = final_loaded_cfg.get('storage_type', "Local")
                                st.session_state.ui_project_name_input = st.session_state.current_project_name
                                st.session_state.ui_project_directory_input = config_dir_to_open
                                initialize_project_data_states()
                                # Removed success message banner when opening a project
                                st.rerun()
                    except yaml.YAMLError:
                        ui_helpers.show_error_message(f"Error parsing the YAML structure of the config file: {full_path_to_config_open}")
                    except Exception as e_open_project:
                        ui_helpers.show_error_message(f"An error occurred while opening project: {e_open_project}")
                        logger.error(f"Error opening project: {e_open_project}", exc_info=True)


    # RIGHT COLUMN: Existing Projects list
    with right_col:
        with st.container(height=300):
            st.subheader("Recent Projects")
            for proj in st.session_state.get('past_projects', []):
                is_active = proj['name'] == st.session_state.current_project_name
                path = os.path.normpath(proj['path'])
                color = "#d4edda" if is_active else "#fff3cd"
                # Use four columns, both sharing the same background style, with reduced spacing
                cols = st.columns([0.7, 0.1, 0.1, 0.1], gap="small")
                bg_style = f"background-color: {color}; padding: 2px; border-radius: 4px; display: block; line-height: 1.2;"
                # Left: project name and path
                with cols[0]:
                    st.markdown(
                        f"<div style='{bg_style} max-width: 400px;'>"
                        f"<strong style=\"color: #000\">{proj['name']}</strong><br>"
                        f"<span style='color: #000; display: inline-block; width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;' title=\"{path}\">{path}</span>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                # cols[1]: open button (icon only, no st.image)
                with cols[1]:
                    if st.button("", icon=":material/open_in_new:", key=f"open_past_{proj['name']}", help="Open"):
                        config_path = proj['path']
                        config_dir = os.path.dirname(config_path)
                        loaded = utils.load_project_config(config_dir, proj['name'])
                        if loaded:
                            st.session_state.current_project_name = proj['name']
                            st.session_state.project_path = config_dir
                            st.session_state.project_config = loaded
                            initialize_project_data_states()
                            db.save_project_index(proj['name'], proj['path'])
                            st.session_state.past_projects = [
                                p for p in db.load_project_index()
                                if p['name'] not in st.session_state.removed_projects
                            ]
                            st.rerun()
                        else:
                            ui_helpers.show_error_message(f"Failed to load project '{proj['name']}' from {config_path}")
                # cols[2]: Remove button (icon only, no st.image)
                with cols[2]:
                    if st.button("", icon=":material/remove_circle:", key=f"remove_past_{proj['name']}", help="Remove"):
                        st.session_state.removed_projects.append(proj['name'])
                        db.delete_project_index(proj['name'])
                        st.session_state.past_projects = [
                            p for p in st.session_state.past_projects
                            if p['name'] not in st.session_state.removed_projects
                        ]
                        st.rerun()
                # cols[3]: Delete button (icon only, no st.image)
                with cols[3]:
                    if st.button("", icon=":material/delete:", key=f"delete_past_{proj['name']}", help="Delete"):
                        # Delete folder
                        try:
                            folder_to_delete = os.path.dirname(path)
                            shutil.rmtree(folder_to_delete)
                            db.delete_project_index(proj['name'])
                        except Exception as e:
                            ui_helpers.show_error_message(f"Failed to delete folder: {e}")
                        # Remove from in-memory list
                        st.session_state.past_projects = [
                            p for p in st.session_state.past_projects if p['name'] != proj['name']
                        ]
                        st.rerun()

    # Removed display of active project and "No active project." message here

    # Removed workflow markdown from here

    # Navigation hint once a project is open (below two-column layout)
    if st.session_state.current_project_name:
        ui_helpers.nav_button(
            "Stay on this page for editing your Reddit and AI Service API Keys, or click here to navigate to Data Management",
            "pages/data_management.py",
            key="nav_to_data_mgmt"
        )

    if st.session_state.current_project_name:
        st.divider()
        st.header("API Keys")
        if st.session_state.project_path:
            # API keys are always stored in the _config.yaml within the project_path
            api_config_for_display = utils.load_project_config(st.session_state.project_path, st.session_state.current_project_name)
            if not api_config_for_display:
                api_config_for_display = st.session_state.project_config # Fallback to session state if load failed

            with st.expander("Reddit API", expanded=False):
                reddit_cfg_api_form = api_config_for_display.get('reddit_api', {})
                client_id_api_form = st.text_input("Client ID", type="password", value=reddit_cfg_api_form.get('client_id', ''), key="api_reddit_client_id_form_home_nav")
                client_secret_api_form = st.text_input("Client Secret", type="password", value=reddit_cfg_api_form.get('client_secret', ''), key="api_reddit_client_secret_form_home_nav")
                user_agent_default_form = f'Python:ThemelyApp_{utils.generate_project_id(st.session_state.current_project_name)}:v0.1 by /u/YourUsername'
                user_agent_api_form = st.text_input("User Agent", value=reddit_cfg_api_form.get('user_agent', user_agent_default_form), key="api_reddit_user_agent_form_home_nav")

            with st.expander("AI Service API", expanded=False):
                ai_provider_default_api_form = api_config_for_display.get('ai_provider', 'OpenAI')
                try:
                    ai_provider_idx_api_keys_form = ["OpenAI", "Gemini"].index(ai_provider_default_api_form)
                except ValueError:
                    ai_provider_idx_api_keys_form = 0 # Default to OpenAI if current value is somehow invalid

                current_ai_provider_val = api_config_for_display.get('ai_provider', 'OpenAI')
                # Ensure the index is valid for the selectbox
                try:
                    current_ai_provider_idx = ["OpenAI", "Gemini"].index(current_ai_provider_val)
                except ValueError:
                    current_ai_provider_idx = 0 # Default to OpenAI if stored value is not in list

                ai_provider_select_api_form = st.selectbox(
                    "Provider",
                    ["OpenAI", "Gemini"],
                    index=current_ai_provider_idx,
                    key="api_ai_provider_select_form_home_nav"
                )

                ai_key_val_api_form = api_config_for_display.get(f'{ai_provider_select_api_form.lower()}_api', {}).get('api_key', '')
                ai_key_input_api_form = st.text_input(
                    "API Key",
                    type="password",
                    value=ai_key_val_api_form,
                    key=f"api_ai_key_input_form_home_nav_{ai_provider_select_api_form}" # Key changes with provider to reset value
                )

            if st.button("Save", key="api_save_keys_button_form_home_nav"):
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

# --- Sidebar Setup ---
st.sidebar.title("Themely")
project_name = st.session_state.get('current_project_name') or "None"
if project_name != "None":
    st.sidebar.success(f"Active project: {project_name}")
else:
    st.sidebar.warning("Active project: None")
st.sidebar.divider()

page_definitions = [
    st.Page(project_setup_page_content, title="Project Setup", default=True),
    st.Page("pages/data_management.py", title="Data Management"),
    st.Page("pages/codebook.py", title="Codebook"),
    st.Page("pages/themes.py", title="Themes"),
    st.Page("pages/analysis.py", title="Analysis"),
]
pg = st.navigation(page_definitions)


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