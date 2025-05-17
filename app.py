# app.py
import streamlit as st
import pandas as pd # Keep for any helpers that might remain or for type hints
import os
import requests
import yaml
from datetime import datetime
import shutil
import subprocess
import platform
import zipfile
import io
import sys

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
# Moved st.set_page_config() to the top.
st.set_page_config(
    layout="wide",
    page_title="Themely - Qualitative Analysis"
)

# Import modules AFTER st.set_page_config, especially those that might use Streamlit commands at import time or early calls.
from modules import auth, utils, ui_helpers
from modules import db

# --- Initialize Database and Global Settings ---
# Now this block runs AFTER st.set_page_config().
if 'db_initialized' not in st.session_state:
    db.init_db()
    st.session_state.db_initialized = True

if 'model_cache_dir' not in st.session_state:
    st.session_state.model_cache_dir = db.load_setting("model_cache_dir") or os.path.join(os.path.expanduser("~"), ".themely_models")
if 'ui_model_cache_dir_input' not in st.session_state: # For the input field
    st.session_state.ui_model_cache_dir_input = st.session_state.model_cache_dir


if 'past_projects' not in st.session_state:
    st.session_state.past_projects = db.load_project_index()
if 'removed_projects' not in st.session_state:
    st.session_state.removed_projects = []
# Filter out any removed names
st.session_state.past_projects = [
    p for p in st.session_state.past_projects
    if p['name'] not in st.session_state.removed_projects
]

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
if 'ui_open_config_file_path_input' not in st.session_state: st.session_state.ui_open_config_file_path_input = ""

# Data-related states that need to be reset when projects change.
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

# --- Local AI Methods Configuration (Static for now) ---
LOCAL_AI_METHODS_CONFIG = {
    "codebook_generation": {
        "default_method": "LLM-Assisted Codebook Generation",
        "options": {
            "LLM-Assisted Codebook Generation": {
                "libraries": ["llama.cpp (Gemma-3-1B-it)"],
                "model_size": "~1B parameters (quantized 4-bit, ~0.8 GB)",
                "min_requirements": "CPU (Modern, AVX2), ~2 GB RAM available",
                "description": "Uses a quantized local Gemma-3-1B-it model to generate code names, descriptions, rationales, and example IDs."
            }
        }
    },
    "coding_with_codebook": {
        "default_method": "Embedding Similarity Matching",
        "options": {
            "Embedding Similarity Matching": {
                "libraries": ["SentenceTransformers (all-MiniLM-L6-v2)"],
                "model_size": "~80MB (MiniLM)",
                "min_requirements": "CPU (Modern), ~500MB RAM available during operation",
                "description": "Generates embeddings for posts and codebook entries, then assigns codes based on cosine similarity above a threshold."
            }
        }
    },
    "grouping_codes_to_themes": {
        "default_method": "Embedding-based Clustering",
        "options": {
            "Embedding-based Clustering": {
                "libraries": ["SentenceTransformers (all-MiniLM-L6-v2)", "Scikit-learn"],
                "model_size": "~80MB (MiniLM) + Scikit-learn (variable)",
                "min_requirements": "CPU (Modern), ~500MB RAM available during operation",
                "description": "Embeds code definitions and clusters them using algorithms like Agglomerative Clustering or K-Means."
            }
        }
    },
    "generating_theme_summaries": {
        "default_method": "Extractive Summarization with References",
        "options": {
            "Extractive Summarization with References": {
                "libraries": ["Sumy (LexRank/LSA) or Gensim (TextRank)"],
                "model_size": "Minimal (algorithm-based)",
                "min_requirements": "CPU, ~200MB RAM available during operation",
                "description": "Selects salient sentences directly from the theme's posts and provides references (unique_app_id, code_ids) for each summarized part."
            }
        }
    }
}


# --- Content for the Home Page (Project Setup) ---
def project_setup_page_content():
    st.title("Themely")
    st.header("Project Setup")
    st.write("Get started by creating a new project or opening an existing one, then enter your Reddit and AI service API keys or configure Local AI.")
    ui_helpers.page_sidebar_info([
        "Create or open a project",
        "Enter Reddit API credentials",
        "Configure AI provider (Cloud or Local)"
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

        if st.session_state.project_action_choice == "New Project":
            st.markdown("Set project name and directory.")
            col_dir_cr, col_sep_cr_ui, col_name_cr = st.columns([0.7, 0.05, 0.25])
            with col_dir_cr:
                dir_input_key_cr = "ui_create_dir_input_key_home_nav"
                ui_dir_val_cr = st.text_input(
                    "Directory:", value=st.session_state.ui_project_directory_input,
                    key=dir_input_key_cr, on_change=lambda: setattr(st.session_state, 'ui_project_directory_input', st.session_state[dir_input_key_cr])
                )
            with col_sep_cr_ui:
                st.markdown(f"<div style='text-align: center; margin-top: 28px; font-size: 1.2em;'>/</div>", unsafe_allow_html=True)
            with col_name_cr:
                name_input_key_cr = "ui_create_name_input_key_home_nav"
                ui_name_val_cr = st.text_input(
                    "Project name:", value=st.session_state.ui_project_name_input,
                    key=name_input_key_cr, on_change=lambda: setattr(st.session_state, 'ui_project_name_input', st.session_state[name_input_key_cr])
                )

            if st.button("Create", key="create_project_button_action_home_nav"):
                dir_to_create_in_val_form = st.session_state.ui_project_directory_input
                name_to_create_val_form = st.session_state.ui_project_name_input
                storage_for_create_val_form = "Local"
                if not dir_to_create_in_val_form or not name_to_create_val_form:
                    ui_helpers.show_error_message("New Project Directory and New Project Name are required.")
                elif not os.path.isabs(dir_to_create_in_val_form):
                    ui_helpers.show_error_message("New Project Directory must be an absolute path.")
                else:
                    project_id = utils.generate_project_id(name_to_create_val_form)
                    project_folder = os.path.join(dir_to_create_in_val_form, project_id)
                    try:
                        os.makedirs(project_folder, exist_ok=True)
                    except Exception as e:
                        ui_helpers.show_error_message(f"Failed to create project folder '{project_folder}': {e}")
                        return

                    if auth.setup_project_storage(
                        name_to_create_val_form,
                        storage_type=storage_for_create_val_form,
                        user_defined_project_dir=project_folder
                    ):
                        st.session_state.current_project_name = name_to_create_val_form
                        st.session_state.project_path = project_folder
                        
                        config_filename_cr = utils.PROJECT_CONFIG_FILE_TEMPLATE.format(project_name_id=project_id)
                        full_config_path_cr = os.path.join(project_folder, config_filename_cr)
                        db.save_project_index(name_to_create_val_form, full_config_path_cr)
                        
                        st.session_state.past_projects = [
                            p for p in db.load_project_index()
                            if p['name'] not in st.session_state.removed_projects
                        ]
                        
                        st.session_state.project_config = utils.load_project_config(project_folder, project_id)
                        
                        if not st.session_state.project_config:
                            ui_helpers.show_error_message("Critical error: Project config could not be loaded after creation.")
                            st.session_state.current_project_name = None
                            st.session_state.project_path = None
                        else:
                            st.session_state.project_config['project_name'] = name_to_create_val_form
                            st.session_state.project_config['project_config_file_directory'] = project_folder
                            st.session_state.project_config['storage_type'] = "Local"
                            st.session_state.project_config['project_id_for_filename'] = project_id
                            utils.save_project_config(project_folder, project_id, st.session_state.project_config)
                            
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

            if st.button("Open", key="open_project_button_action_home_nav"):
                full_path_to_config_open = st.session_state.ui_open_config_file_path_input
                if not full_path_to_config_open or not os.path.isabs(full_path_to_config_open) or not os.path.isfile(full_path_to_config_open):
                    ui_helpers.show_error_message(f"Invalid path or file not found: {full_path_to_config_open}")
                else:
                    try:
                        config_dir_to_open = os.path.dirname(full_path_to_config_open)
                        base_filename = os.path.basename(full_path_to_config_open)
                        
                        project_id_from_filename = None
                        if base_filename.endswith(f"_{utils.PROJECT_CONFIG_FILE_TEMPLATE.split('_', 1)[1]}"):
                            project_id_from_filename = base_filename.rsplit('_', 1)[0]
                        
                        loaded_cfg = utils.load_project_config(config_dir_to_open, project_id_from_filename)

                        if not loaded_cfg:
                            with open(full_path_to_config_open, 'r') as f_direct_open:
                                temp_cfg_open = yaml.safe_load(f_direct_open)
                            project_name_from_cfg_content = temp_cfg_open.get('project_name')
                            if not project_name_from_cfg_content:
                                ui_helpers.show_error_message("Project name not found within the config file. Cannot open by direct path if ID inference fails.")
                                return
                            project_id_from_name_in_cfg = utils.generate_project_id(project_name_from_cfg_content)
                            loaded_cfg = utils.load_project_config(config_dir_to_open, project_id_from_name_in_cfg)

                        if not loaded_cfg:
                            ui_helpers.show_error_message(f"Error processing config. File may be corrupted, structure incorrect, or project name/ID mismatch.")
                        else:
                            st.session_state.project_config = loaded_cfg
                            st.session_state.current_project_name = loaded_cfg.get('project_name')
                            st.session_state.project_path = config_dir_to_open
                            
                            db.save_project_index(st.session_state.current_project_name, full_path_to_config_open)
                            st.session_state.past_projects = [
                                p for p in db.load_project_index()
                                if p['name'] not in st.session_state.removed_projects
                            ]
                            
                            st.session_state.project_config['project_config_file_directory'] = config_dir_to_open
                            st.session_state.project_config.setdefault('storage_type', "Local")
                            if 'project_id_for_filename' not in st.session_state.project_config:
                                st.session_state.project_config['project_id_for_filename'] = utils.generate_project_id(st.session_state.current_project_name)

                            st.session_state.ui_project_name_input = st.session_state.current_project_name
                            st.session_state.ui_project_directory_input = os.path.dirname(st.session_state.project_path)

                            initialize_project_data_states()
                            st.rerun()
                    except yaml.YAMLError:
                        ui_helpers.show_error_message(f"Error parsing the YAML structure of the config file: {full_path_to_config_open}")
                    except Exception as e_open_project:
                        ui_helpers.show_error_message(f"An error occurred while opening project: {e_open_project}")
                        logger.error(f"Error opening project: {e_open_project}", exc_info=True)

    with right_col:
        with st.container(height=300):
            st.subheader("Recent Projects")
            if not st.session_state.get('past_projects', []):
                st.info("No recent projects found. Create or open a project to get started.")

            for proj in st.session_state.get('past_projects', []):
                is_active = proj['name'] == st.session_state.current_project_name
                config_file_full_path_for_proj = os.path.normpath(proj['path'])
                project_base_path_for_proj = os.path.dirname(config_file_full_path_for_proj)

                color = "#d4edda" if is_active else "#fff3cd"
                cols = st.columns([0.7, 0.1, 0.1, 0.1], gap="small")
                bg_style = f"background-color: {color}; padding: 2px; border-radius: 4px; display: block; line-height: 1.2;"

                with cols[0]:
                    st.markdown(
                        f"<div style='{bg_style} max-width: 400px;'>"
                        f"<strong style=\"color: #000\">{proj['name']}</strong><br>"
                        f"<span style='color: #000; display: inline-block; width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;' title=\"{project_base_path_for_proj}\">{project_base_path_for_proj}</span>"
                        "</div>",
                        unsafe_allow_html=True
                    )
                with cols[1]:
                    if st.button("", icon=":material/open_in_new:", key=f"open_past_{proj['name']}", help="Open this project"):
                        config_path_to_open = proj['path']
                        config_dir = os.path.dirname(config_path_to_open)
                        project_id_to_load = utils.generate_project_id(proj['name'])

                        loaded_cfg = utils.load_project_config(config_dir, project_id_to_load)
                        if loaded_cfg:
                            st.session_state.current_project_name = proj['name']
                            st.session_state.project_path = config_dir
                            st.session_state.project_config = loaded_cfg
                            initialize_project_data_states()
                            db.save_project_index(proj['name'], config_path_to_open)
                            st.session_state.past_projects = [p for p in db.load_project_index() if p['name'] not in st.session_state.removed_projects]
                            st.rerun()
                        else:
                            ui_helpers.show_error_message(f"Failed to load project '{proj['name']}' from {config_path_to_open}")
                with cols[2]:
                    if st.button("", icon=":material/remove_circle:", key=f"remove_past_{proj['name']}", help="Remove from this list (does not delete files)"):
                        st.session_state.removed_projects.append(proj['name'])
                        db.delete_project_index(proj['name'])
                        st.session_state.past_projects = [p for p in st.session_state.past_projects if p['name'] not in st.session_state.removed_projects]
                        if st.session_state.current_project_name == proj['name']:
                            st.session_state.current_project_name = None
                            st.session_state.project_path = None
                            st.session_state.project_config = {}
                        st.rerun()
                with cols[3]:
                    delete_key = f"delete_past_{proj['name']}_{utils.generate_project_id(proj['name'])}"
                    if st.button("", icon=":material/delete:", key=delete_key, help="Delete project files and folder"):
                        st.session_state.confirm_delete_project_name = proj['name']
                        st.session_state.confirm_delete_project_path_to_config = proj['path']
                        st.rerun() # Rerun to show confirmation dialog

    if st.session_state.get('confirm_delete_project_name'):
        proj_name_to_delete = st.session_state.confirm_delete_project_name
        config_file_path_to_delete = st.session_state.confirm_delete_project_path_to_config
        project_folder_to_delete = os.path.dirname(config_file_path_to_delete)

        st.error(f"**Confirm Deletion:** Are you sure you want to permanently delete all files and the folder for project '{proj_name_to_delete}' located at `{project_folder_to_delete}`? This action cannot be undone.")
        confirm_col1, confirm_col2, confirm_col3_spacer = st.columns([1,1,3])
        if confirm_col1.button("YES, DELETE PROJECT FILES", key="confirm_delete_files_btn_dialog", type="primary"):
            try:
                shutil.rmtree(project_folder_to_delete)
                db.delete_project_index(proj_name_to_delete)
                st.session_state.removed_projects.append(proj_name_to_delete)
                st.session_state.past_projects = [
                    p for p in st.session_state.past_projects
                    if p['name'] != proj_name_to_delete
                ]
                if st.session_state.current_project_name == proj_name_to_delete:
                    st.session_state.current_project_name = None
                    st.session_state.project_path = None
                    st.session_state.project_config = {}
                ui_helpers.show_success_message(f"Project '{proj_name_to_delete}' and its folder have been deleted.")
            except Exception as e:
                ui_helpers.show_error_message(f"Failed to delete project folder '{project_folder_to_delete}': {e}")
            st.session_state.confirm_delete_project_name = None # Clear confirmation state
            st.session_state.confirm_delete_project_path_to_config = None
            st.rerun()
        if confirm_col2.button("CANCEL", key="cancel_delete_files_btn_dialog"):
            st.session_state.confirm_delete_project_name = None
            st.session_state.confirm_delete_project_path_to_config = None
            st.rerun()


    if st.session_state.current_project_name:
        ui_helpers.nav_button(
            "Stay on this page for editing your Reddit and AI Service API Keys, or click here to navigate to Data Management",
            "pages/data_management.py",
            key="nav_to_data_mgmt_home"
        )

    if st.session_state.current_project_name:
        st.divider()
        st.header("API Keys & Local AI Configuration")

        if st.session_state.project_path:
            project_id_for_keys = st.session_state.project_config.get('project_id_for_filename') or utils.generate_project_id(st.session_state.current_project_name)
            api_config_for_display = utils.load_project_config(st.session_state.project_path, project_id_for_keys)

            if not api_config_for_display:
                api_config_for_display = st.session_state.project_config

            with st.expander("Reddit API", expanded=False):
                reddit_cfg_api_form = api_config_for_display.get('reddit_api', {})
                client_id_api_form = st.text_input("Client ID", type="password", value=reddit_cfg_api_form.get('client_id', ''), key="api_reddit_client_id_form_home_nav_exp")
                client_secret_api_form = st.text_input("Client Secret", type="password", value=reddit_cfg_api_form.get('client_secret', ''), key="api_reddit_client_secret_form_home_nav_exp")
                user_agent_default_form = f'Python:ThemelyApp_{project_id_for_keys}:v0.1 by /u/YourUsername'
                user_agent_api_form = st.text_input("User Agent", value=reddit_cfg_api_form.get('user_agent', user_agent_default_form), key="api_reddit_user_agent_form_home_nav_exp")

            with st.expander("AI Service Configuration", expanded=True):
                ai_provider_options = ["OpenAI", "Gemini", "Local"]
                ai_provider_default_api_form = api_config_for_display.get('ai_provider', 'OpenAI')
                try:
                    ai_provider_idx_api_keys_form = ai_provider_options.index(ai_provider_default_api_form)
                except ValueError:
                    ai_provider_idx_api_keys_form = 0

                ai_provider_select_api_form = st.selectbox(
                    "Provider",
                    ai_provider_options,
                    index=ai_provider_idx_api_keys_form,
                    key="api_ai_provider_select_form_home_nav_exp"
                )

                ai_key_input_api_form = ""
                local_ai_settings_to_save = {}

                if ai_provider_select_api_form in ["OpenAI", "Gemini"]:
                    ai_key_val_api_form = api_config_for_display.get(f'{ai_provider_select_api_form.lower()}_api', {}).get('api_key', '')
                    ai_key_input_api_form = st.text_input(
                        "API Key",
                        type="password",
                        value=ai_key_val_api_form,
                        key=f"api_ai_key_input_form_home_nav_{ai_provider_select_api_form}_exp"
                    )
                elif ai_provider_select_api_form == "Local":
                    st.markdown("---")
                    st.subheader("Local AI Setup")
                    st.write("Themely will use local models for AI tasks. No data is sent to external services.")

                    st.markdown("**Model Cache Directory**")
                    st.write("Specify a directory on your computer where downloaded models will be stored.")

                    cache_dir_col1, cache_dir_col2 = st.columns([3,1])
                    with cache_dir_col1:
                        st.session_state.ui_model_cache_dir_input = st.text_input(
                            "Cache Directory Path:",
                            value=st.session_state.ui_model_cache_dir_input, # Use the dedicated UI state variable
                            key="global_model_cache_dir_input_app_exp",
                            help="e.g., C:/Users/YourName/.themely_models or /home/yourname/.themely_models"
                        )
                    with cache_dir_col2:
                        st.write("")
                        st.write("")
                        if st.button("Save Cache Path", key="save_global_model_cache_dir_btn_app_exp"):
                            new_cache_path = st.session_state.ui_model_cache_dir_input.strip()
                            if os.path.isabs(new_cache_path):
                                try:
                                    os.makedirs(new_cache_path, exist_ok=True)
                                    db.save_setting("model_cache_dir", new_cache_path)
                                    st.session_state.model_cache_dir = new_cache_path
                                    ui_helpers.show_success_message(f"Model cache directory saved: {new_cache_path}")
                                    st.rerun() # Rerun to reflect change in sidebar if needed
                                except Exception as e:
                                    ui_helpers.show_error_message(f"Error setting cache directory: {e}")
                            else:
                                ui_helpers.show_error_message("Cache directory must be an absolute path.")


                    st.markdown("---")
                    st.markdown("**Local AI Methods & Models:**")
                    st.caption("The following methods and models/libraries will be used for AI-assisted tasks. These run entirely on your CPU.")

                    current_local_ai_config_from_project = api_config_for_display.get('local_ai_config', {})

                    for task_key, task_config in LOCAL_AI_METHODS_CONFIG.items():
                        task_name = task_key.replace("_", " ").title()
                        st.markdown(f"**{task_name}:**")

                        selected_method_key_ui = f"local_method_{task_key}_ui_exp"
                        default_method_for_task = current_local_ai_config_from_project.get(task_key, {}).get("method", task_config["default_method"])

                        if default_method_for_task not in task_config["options"]:
                            default_method_for_task = task_config["default_method"]

                        try:
                            method_options = list(task_config["options"].keys())
                            current_method_idx = method_options.index(default_method_for_task)
                        except ValueError:
                            method_options = [task_config["default_method"]]
                            current_method_idx = 0

                        selected_method = st.selectbox(
                            f"Method for {task_name}:",
                            options=method_options,
                            index=current_method_idx,
                            key=selected_method_key_ui
                        )
                        # --- Insert LLaMA download logic for codebook_generation ---
                        if task_key == "codebook_generation":
                            st.markdown("**Download Local Gemma Model**")
                            model_filename = "gemma-3-1b-pt-q4_0.gguf"
                            model_path = os.path.join(st.session_state.model_cache_dir, model_filename)
                            download_url = "https://huggingface.co/google/gemma-3-1b-pt-qat-q4_0-gguf/resolve/main/gemma-3-1b-pt-q4_0.gguf"

                            if os.path.exists(model_path):
                                st.success("Gemma model ready.")
                            else:
                                if st.button("Download Gemma-3-1B-pt model", key="download_gemma_model_task"):
                                    download_gemma_model_dialog()
                        method_details = task_config["options"][selected_method]
                        st.markdown(f"*   **Uses:** `{', '.join(method_details['libraries'])}`")
                        st.markdown(f"*   **Approx. Model Size(s):** `{method_details['model_size']}` (downloads to cache directory)")
                        st.markdown(f"*   **Min. System Requirements:** `{method_details['min_requirements']}`")
                        st.markdown(f"*   *{method_details['description']}*")
                        st.markdown("---")
                        local_ai_settings_to_save[task_key] = {"method": selected_method}


            if st.button("Save Configuration", key="api_save_keys_button_form_home_nav_exp"):
                r_keys_form = {"client_id": client_id_api_form, "client_secret": client_secret_api_form, "user_agent": user_agent_api_form}
                gai_keys_to_store = {}
                if ai_provider_select_api_form in ["OpenAI", "Gemini"]:
                    gai_keys_to_store = {"api_key": ai_key_input_api_form}
                elif ai_provider_select_api_form == "Local":
                    gai_keys_to_store = local_ai_settings_to_save

                valid_reddit = all(r_keys_form.values())
                valid_gai = True
                if ai_provider_select_api_form in ["OpenAI", "Gemini"]:
                    valid_gai = bool(gai_keys_to_store.get("api_key"))

                if not valid_reddit:
                    ui_helpers.show_warning_message("Reddit API fields cannot be empty. Keys not saved.")
                elif not valid_gai and ai_provider_select_api_form != "Local":
                    ui_helpers.show_warning_message(f"{ai_provider_select_api_form} API Key cannot be empty. Keys not saved.")
                elif auth.validate_api_keys(r_keys_form, "Reddit") and \
                     auth.validate_api_keys(gai_keys_to_store if ai_provider_select_api_form != "Local" else {}, ai_provider_select_api_form):
                    project_id_for_saving_config = st.session_state.project_config.get('project_id_for_filename') or utils.generate_project_id(st.session_state.current_project_name)
                    if auth.store_api_keys(r_keys_form, gai_keys_to_store, ai_provider_select_api_form):
                        st.session_state.project_config = utils.load_project_config(st.session_state.project_path, project_id_for_saving_config)
                        ui_helpers.show_success_message("API keys and AI configuration saved successfully to project configuration.")
                        st.rerun()

st.sidebar.title("Themely")
project_name_sidebar = st.session_state.get('current_project_name') or "None"
if project_name_sidebar != "None":
    st.sidebar.success(f"Active project: {project_name_sidebar}")
else:
    st.sidebar.warning("Active project: None")
st.sidebar.divider()
# Display the model cache directory from session state, which is loaded from DB
st.sidebar.info(f"Model Cache: {st.session_state.get('model_cache_dir', 'Not Set')}")



# --- Dialog: Download Gemma Model ---
@st.dialog("Download Gemma Model", width="medium")
def download_gemma_model_dialog():
    st.markdown("#### Download Gemma‑3‑1B‑pt QAT (Q4_0)")
    st.markdown("""
1. Log in to **Hugging Face** and accept the Gemma licence: <https://huggingface.co/google/gemma-3-1b-pt-qat-q4_0-gguf>.
2. Create a *read‑only* access token (Settings → **Access Tokens**) and paste it below.
The script will fetch the quantised GGUF into your local model cache.
    """)

    cache_dir = st.session_state.model_cache_dir
    token = st.text_input("Hugging Face Token", type="password", key="hf_token_input")

    model_filename = "gemma-3-1b-pt-q4_0.gguf"
    model_path = os.path.join(cache_dir, model_filename)
    model_url = f"https://huggingface.co/google/gemma-3-1b-pt-qat-q4_0-gguf/resolve/main/{model_filename}"

    if os.path.exists(model_path):
        st.success("Gemma model already present.")
    else:
        if st.button("Download model", key="hf_start_download_gemma"):
            progress_bar = st.progress(0.0)
            os.makedirs(cache_dir, exist_ok=True)
            with st.spinner("Downloading…"):
                try:
                    headers = {"Authorization": f"Bearer {token}"} if token else {}
                    r = requests.get(model_url, headers=headers, stream=True)
                    r.raise_for_status()
                    total = int(r.headers.get("Content-Length", 0))
                    downloaded = 0
                    with open(model_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total:
                                progress_bar.progress(min(downloaded / total, 1.0))
                    st.success("Gemma model downloaded successfully!")
                except Exception as e:
                    st.error(f"Download failed: {e}")

    if st.button("Cancel", key="hf_cancel_gemma"):
        st.rerun()

page_definitions = [
    st.Page(project_setup_page_content, title="Project Setup", default=True, icon="🛠️"),
    st.Page("pages/data_management.py", title="Data Management", icon="💾"),
    st.Page("pages/codebook.py", title="Codebook", icon="📖"),
    st.Page("pages/themes.py", title="Themes", icon="📊"),
    st.Page("pages/analysis.py", title="Analysis", icon="📝"),
]
pg = st.navigation(page_definitions)

pg.run()

if st.session_state.current_project_name:
    project_id_for_recovery = st.session_state.project_config.get('project_id_for_filename') or utils.generate_project_id(st.session_state.current_project_name)
    if not st.session_state.project_path and st.session_state.project_config.get('project_config_file_directory'):
        logger.warning("project_path missing, attempting recovery from project_config['project_config_file_directory']")
        st.session_state.project_path = st.session_state.project_config['project_config_file_directory']
    elif st.session_state.project_path and \
         st.session_state.project_config.get('project_config_file_directory') != st.session_state.project_path:
        logger.warning(f"project_path ('{st.session_state.project_path}') and config's dir ('{st.session_state.project_config.get('project_config_file_directory')}') mismatch. Attempting to reload config from project_path.")
        loaded_cfg_recovery_main = utils.load_project_config(st.session_state.project_path, project_id_for_recovery)
        if loaded_cfg_recovery_main:
            st.session_state.project_config = loaded_cfg_recovery_main
            st.session_state.project_config['project_config_file_directory'] = st.session_state.project_path
            st.session_state.project_config.setdefault('storage_type', "Local")
            if 'project_id_for_filename' not in st.session_state.project_config:
                 st.session_state.project_config['project_id_for_filename'] = project_id_for_recovery
            utils.save_project_config(st.session_state.project_path, project_id_for_recovery, st.session_state.project_config)
        else:
            logger.error(f"Recovery failed for {st.session_state.current_project_name} due to config load error from project_path.")