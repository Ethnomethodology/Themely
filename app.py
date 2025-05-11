# app.py
import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import yaml
from datetime import datetime
import json # For formatting batch data to AI

from modules import auth, reddit_api, ai_services, data_management, ui_helpers, utils

st.set_page_config(layout="wide", page_title="Qualitative Thematic Analysis Tool")
logger = utils.setup_logger("app")

# --- Initialize Session State (ensure all new states are here) ---
if 'project_action_choice' not in st.session_state: st.session_state.project_action_choice = "Create New Project"
if 'current_project_name' not in st.session_state: st.session_state.current_project_name = None
if 'project_config' not in st.session_state: st.session_state.project_config = {}
if 'project_path' not in st.session_state: st.session_state.project_path = None
if 'active_tab' not in st.session_state: st.session_state.active_tab = "Project Setup & API Config"

if 'ui_project_directory_input' not in st.session_state: st.session_state.ui_project_directory_input = os.path.expanduser("~")
if 'ui_project_name_input' not in st.session_state: st.session_state.ui_project_name_input = ""
if 'ui_storage_type_select_create' not in st.session_state: st.session_state.ui_storage_type_select_create = "Local"
if 'ui_open_config_file_path_input' not in st.session_state: st.session_state.ui_open_config_file_path_input = ""

if 'selected_download_files_info' not in st.session_state: st.session_state.selected_download_files_info = {}
if 'combined_data_for_view_creation' not in st.session_state: st.session_state.combined_data_for_view_creation = pd.DataFrame()
if 'search_term_combined_data_tab2' not in st.session_state: st.session_state.search_term_combined_data_tab2 = ""
if 'redact_confirm_tab2_combined' not in st.session_state: st.session_state.redact_confirm_tab2_combined = False

if 'selected_project_views_info_tab3' not in st.session_state: st.session_state.selected_project_views_info_tab3 = {}
if 'data_for_coding_tab3' not in st.session_state: st.session_state.data_for_coding_tab3 = pd.DataFrame()
if 'col_for_coding_tab3' not in st.session_state: st.session_state.col_for_coding_tab3 = "text"
if 'id_col_for_coding_tab3' not in st.session_state: st.session_state.id_col_for_coding_tab3 = "unique_app_id"
if 'search_term_coding_data_tab3' not in st.session_state: st.session_state.search_term_coding_data_tab3 = ""
if 'currently_selected_view_filepaths_for_saving_tab3' not in st.session_state: st.session_state.currently_selected_view_filepaths_for_saving_tab3 = []
if 'prompt_save_combined_as_new_view_tab3' not in st.session_state: st.session_state.prompt_save_combined_as_new_view_tab3 = False
if 'ai_batch_size_tab3' not in st.session_state: st.session_state.ai_batch_size_tab3 = 10


def initialize_project_data_states():
    st.session_state.selected_download_files_info = {}
    st.session_state.combined_data_for_view_creation = pd.DataFrame()
    st.session_state.selected_project_views_info_tab3 = {}
    st.session_state.data_for_coding_tab3 = pd.DataFrame()
    st.session_state.currently_selected_view_filepaths_for_saving_tab3 = []
    logger.info("Project-specific data session states initialized/reset.")

def robust_list_to_comma_string(code_input):
    if isinstance(code_input, list):
        return ", ".join(str(c).strip() for c in code_input if str(c).strip())
    if pd.isna(code_input) or not str(code_input).strip(): return ""
    return str(code_input).strip()

def robust_comma_string_to_list(code_str):
    if pd.isna(code_str) or not str(code_str).strip(): return []
    if isinstance(code_str, list): return [str(c).strip() for c in code_str if str(c).strip()]
    return [c.strip() for c in str(code_str).split(',') if c.strip()]

def load_and_combine_selected_downloads_for_view():
    combined_df_list = []
    selected_file_infos = [ data for data in st.session_state.selected_download_files_info.values() if data.get("selected", False) ]
    if not selected_file_infos:
        st.session_state.combined_data_for_view_creation = pd.DataFrame()
        return
    for file_info in selected_file_infos:
        df_single = data_management.load_data_from_specific_file(file_info["filepath"])
        if df_single is not None:
            df_single["Source File"] = os.path.basename(file_info["filepath"])
            combined_df_list.append(df_single)
    if combined_df_list:
        concatenated_df = pd.concat(combined_df_list, ignore_index=True)
        if 'unique_app_id' in concatenated_df.columns:
            concatenated_df.drop_duplicates(subset=['unique_app_id'], keep='first', inplace=True)
        elif 'id' in concatenated_df.columns:
             concatenated_df.drop_duplicates(subset=['id'], keep='first', inplace=True)
        else: concatenated_df.drop_duplicates(keep='first', inplace=True)
        st.session_state.combined_data_for_view_creation = concatenated_df
    else: st.session_state.combined_data_for_view_creation = pd.DataFrame()

def load_and_combine_selected_views_for_coding():
    combined_df_list_views = []
    st.session_state.currently_selected_view_filepaths_for_saving_tab3 = []
    selected_view_infos = [ data for data in st.session_state.selected_project_views_info_tab3.values() if data.get("selected", False) ]
    if not selected_view_infos:
        st.session_state.data_for_coding_tab3 = pd.DataFrame()
        return
    for view_info in selected_view_infos:
        csv_path = view_info["metadata"].get("csv_filepath")
        if csv_path and os.path.exists(csv_path):
            df_single_view = data_management.load_data_from_specific_file(csv_path)
            if df_single_view is not None:
                df_single_view["Source View"] = view_info["metadata"].get("view_name", os.path.basename(csv_path))
                combined_df_list_views.append(df_single_view)
                st.session_state.currently_selected_view_filepaths_for_saving_tab3.append(csv_path)
        else: logger.warning(f"CSV filepath missing or invalid for view: {view_info['metadata'].get('view_name')}")
    if combined_df_list_views:
        concatenated_df = pd.concat(combined_df_list_views, ignore_index=True)
        if 'unique_app_id' in concatenated_df.columns:
            concatenated_df.drop_duplicates(subset=['unique_app_id'], keep='first', inplace=True)
        elif 'id' in concatenated_df.columns:
            concatenated_df.drop_duplicates(subset=['id'], keep='first', inplace=True)
        else: concatenated_df.drop_duplicates(keep='first', inplace=True)
        if 'Codes' not in concatenated_df.columns:
            concatenated_df['Codes'] = "" 
        else:
            concatenated_df['Codes'] = concatenated_df['Codes'].apply(robust_list_to_comma_string)
        st.session_state.data_for_coding_tab3 = concatenated_df
    else: st.session_state.data_for_coding_tab3 = pd.DataFrame()

st.title("Qualitative Thematic Analysis for Reddit Data")
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Project Setup & API Config", "2. Data Management & View Creation",
    "3. AI Coding on Views", "4. Analysis & Visualization"
])

with tab1:
    # ... (Tab 1 code - no changes from previous full version for this fix)
    st.header("Project Setup")
    project_action_key = "project_action_radio_main_v10" 
    project_action_index = 0 if st.session_state.get('project_action_choice', "Create New Project") == "Create New Project" else 1
    
    project_action = st.radio(
        "Choose an action:", ("Create New Project", "Open Existing Project"),
        index=project_action_index,
        key=project_action_key, horizontal=True,
        on_change=lambda: setattr(st.session_state, 'project_action_choice', st.session_state[project_action_key])
    )
    st.session_state.project_action_choice = st.session_state[project_action_key] 

    if st.session_state.project_action_choice == "Create New Project":
        st.subheader("Create New Project")
        st.markdown("Specify the **directory** for the new project's configuration file and data, and the **project name**.")
        col_dir_cr_v10, col_sep_cr_ui_v10, col_name_cr_v10 = st.columns([0.7, 0.05, 0.25])
        with col_dir_cr_v10:
            dir_input_key_cr_v10 = "ui_create_dir_input_key_v10"
            ui_dir_val_cr_v10 = st.text_input(
                "New Project Directory (Absolute Path):", value=st.session_state.ui_project_directory_input, 
                key=dir_input_key_cr_v10, on_change=lambda: setattr(st.session_state, 'ui_project_directory_input', st.session_state[dir_input_key_cr_v10])
            )
            st.session_state.ui_project_directory_input = ui_dir_val_cr_v10
        with col_sep_cr_ui_v10: st.markdown(f"<div style='text-align: center; margin-top: 28px; font-size: 1.2em;'>/</div>", unsafe_allow_html=True)
        with col_name_cr_v10:
            name_input_key_cr_v10 = "ui_create_name_input_key_v10"
            ui_name_val_cr_v10 = st.text_input(
                "New Project Name:", value=st.session_state.ui_project_name_input, 
                key=name_input_key_cr_v10, on_change=lambda: setattr(st.session_state, 'ui_project_name_input', st.session_state[name_input_key_cr_v10])
            )
            st.session_state.ui_project_name_input = ui_name_val_cr_v10

        if st.session_state.ui_project_directory_input and st.session_state.ui_project_name_input:
            sanitized_name_cr_preview_v10 = utils.generate_project_id(st.session_state.ui_project_name_input)
            config_filename_cr_preview_v10 = utils.PROJECT_CONFIG_FILE_TEMPLATE.format(project_name_id=sanitized_name_cr_preview_v10)
            full_config_path_cr_preview_v10 = os.path.join(st.session_state.ui_project_directory_input, config_filename_cr_preview_v10)
            st.caption(f"Config file will be approx: `{os.path.normpath(full_config_path_cr_preview_v10)}`")
        
        storage_options_create_v10 = ["Local", "Google Drive", "Dropbox", "OneDrive"]
        storage_select_key_cr_form_v10 = "ui_storage_type_select_create_key_form_v10"
        try: current_storage_idx_cr_form_v10 = storage_options_create_v10.index(st.session_state.ui_storage_type_select_create)
        except ValueError: current_storage_idx_cr_form_v10 = 0
        selected_storage_cr_form_v10 = st.selectbox(
            "Select Storage Type for New Project:", storage_options_create_v10, index=current_storage_idx_cr_form_v10,
            key=storage_select_key_cr_form_v10,
            on_change=lambda: setattr(st.session_state, 'ui_storage_type_select_create', st.session_state[storage_select_key_cr_form_v10])
        )
        st.session_state.ui_storage_type_select_create = selected_storage_cr_form_v10
        if selected_storage_cr_form_v10 != "Local":
            st.info(f"For **{selected_storage_cr_form_v10}**: Data will be on the cloud. Config file in the directory specified above.")

        if st.button("Create New Project", key="create_project_button_action_v10"):
            dir_to_create_in_val_form_v10 = st.session_state.ui_project_directory_input
            name_to_create_val_form_v10 = st.session_state.ui_project_name_input
            storage_for_create_val_form_v10 = st.session_state.ui_storage_type_select_create
            if not dir_to_create_in_val_form_v10 or not name_to_create_val_form_v10:
                ui_helpers.show_error_message("New Project Directory and New Project Name are required.")
            elif not os.path.isabs(dir_to_create_in_val_form_v10):
                ui_helpers.show_error_message("New Project Directory must be an absolute path.")
            else:
                if auth.setup_project_storage(name_to_create_val_form_v10, storage_for_create_val_form_v10, user_defined_config_dir=dir_to_create_in_val_form_v10):
                    st.session_state.current_project_name = name_to_create_val_form_v10
                    st.session_state.project_path = dir_to_create_in_val_form_v10
                    st.session_state.project_config = utils.load_project_config(dir_to_create_in_val_form_v10, name_to_create_val_form_v10)
                    if 'project_config_file_directory' not in st.session_state.project_config or \
                       st.session_state.project_config['project_config_file_directory'] != dir_to_create_in_val_form_v10:
                        st.session_state.project_config['project_config_file_directory'] = dir_to_create_in_val_form_v10
                        utils.save_project_config(dir_to_create_in_val_form_v10, name_to_create_val_form_v10, st.session_state.project_config)
                    initialize_project_data_states() 
                    ui_helpers.show_success_message(f"Project '{name_to_create_val_form_v10}' created successfully!")
                    st.rerun()
                else: ui_helpers.show_error_message("Failed to create project.")

    elif st.session_state.project_action_choice == "Open Existing Project":
        st.subheader("Open Existing Project")
        open_config_path_key_v10 = "ui_open_config_file_path_input_key_v10"
        config_file_path_open_val_v10 = st.text_input(
            "Absolute Path to Project Configuration File (`..._config.yaml`):",
            value=st.session_state.ui_open_config_file_path_input, key=open_config_path_key_v10,
            on_change=lambda: setattr(st.session_state, 'ui_open_config_file_path_input', st.session_state[open_config_path_key_v10])
        )
        st.session_state.ui_open_config_file_path_input = config_file_path_open_val_v10
        if st.button("Open Project", key="open_project_button_action_v10"):
            full_path_to_config_open_v10 = st.session_state.ui_open_config_file_path_input
            if not full_path_to_config_open_v10 or not os.path.isabs(full_path_to_config_open_v10) or not os.path.isfile(full_path_to_config_open_v10):
                ui_helpers.show_error_message(f"Invalid path or file not found: {full_path_to_config_open_v10}")
            else:
                try:
                    config_dir_to_open_v10 = os.path.dirname(full_path_to_config_open_v10)
                    with open(full_path_to_config_open_v10, 'r') as f_direct_v10: direct_loaded_cfg_open_v10 = yaml.safe_load(f_direct_v10)
                    if not direct_loaded_cfg_open_v10 or not isinstance(direct_loaded_cfg_open_v10, dict):
                        ui_helpers.show_error_message(f"Invalid or empty config file: {full_path_to_config_open_v10}")
                    else:
                        project_name_from_cfg_v10 = direct_loaded_cfg_open_v10.get('project_name')
                        if not project_name_from_cfg_v10: ui_helpers.show_error_message("Project name not found within the config file.")
                        else:
                            final_loaded_cfg_v10 = utils.load_project_config(config_dir_to_open_v10, project_name_from_cfg_v10)
                            if not final_loaded_cfg_v10: ui_helpers.show_error_message(f"Error processing config for '{project_name_from_cfg_v10}'.")
                            else:
                                st.session_state.project_config = final_loaded_cfg_v10
                                st.session_state.current_project_name = final_loaded_cfg_v10.get('project_name')
                                st.session_state.project_path = config_dir_to_open_v10
                                st.session_state.project_config['project_config_file_directory'] = config_dir_to_open_v10
                                st.session_state.ui_project_name_input = st.session_state.current_project_name
                                st.session_state.ui_project_directory_input = config_dir_to_open_v10
                                st.session_state.ui_storage_type_select_create = final_loaded_cfg_v10.get('storage_type', "Local")
                                initialize_project_data_states()
                                ui_helpers.show_success_message(f"Project '{st.session_state.current_project_name}' opened successfully!")
                                st.rerun()
                except Exception as e_open_project_v10: ui_helpers.show_error_message(f"An error occurred while opening project: {e_open_project_v10}")
    
    st.markdown("---")
    if st.session_state.current_project_name and st.session_state.project_path:
        active_project_config_dir_info_disp = st.session_state.project_path
        active_storage_type_info_disp = st.session_state.project_config.get('storage_type', 'N/A')
        st.subheader("Current Active Project")
        st.success(f"**{st.session_state.current_project_name}**")
        st.caption(f"Storage Type: {active_storage_type_info_disp}")
        st.caption(f"Project Config File Directory: `{os.path.normpath(active_project_config_dir_info_disp)}`")
        if active_storage_type_info_disp == "Local":
             st.caption(f"Local Data Files also in this directory.")
        elif active_storage_type_info_disp != "N/A":
             cloud_data_path_conceptual_disp = st.session_state.project_config.get(
                 'project_path_on_cloud_conceptual', 
                 f"{active_storage_type_info_disp}_root/{utils.generate_project_id(st.session_state.current_project_name)}"
             )
             st.caption(f"Cloud Data (Conceptual Path): `{cloud_data_path_conceptual_disp}`")

        if active_storage_type_info_disp == "OneDrive": 
            if st.button("Re-authenticate OneDrive", key="reauth_onedrive_active_project_key_v10"): 
                st.session_state.force_onedrive_reauth = True
                if auth.setup_project_storage(st.session_state.current_project_name, "OneDrive", user_defined_config_dir=st.session_state.project_path):
                     st.session_state.project_config = utils.load_project_config(st.session_state.project_path, st.session_state.current_project_name)
                     ui_helpers.show_success_message("OneDrive re-authentication process completed.")
                else: ui_helpers.show_error_message("OneDrive re-authentication process failed or was cancelled.")
                if 'force_onedrive_reauth' in st.session_state: del st.session_state['force_onedrive_reauth']
                st.rerun()
    else:
        st.info("No project is currently active. Create a new project or open an existing one.")

    st.divider()
    st.header("API Key Management")
    if st.session_state.current_project_name and st.session_state.project_path:
        api_config_for_display = utils.load_project_config(st.session_state.project_path, st.session_state.current_project_name)
        if not api_config_for_display: api_config_for_display = st.session_state.project_config

        with st.expander("Reddit API Credentials", expanded=False):
            reddit_cfg_api_form = api_config_for_display.get('reddit_api', {})
            client_id_api_form = st.text_input("Reddit Client ID", type="password", value=reddit_cfg_api_form.get('client_id', ''), key="api_reddit_client_id_form_v10")
            client_secret_api_form = st.text_input("Reddit Client Secret", type="password", value=reddit_cfg_api_form.get('client_secret', ''), key="api_reddit_client_secret_form_v10")
            user_agent_default_form = f'Python:ThemelyApp_{utils.generate_project_id(st.session_state.current_project_name)}:v0.1 by /u/YourUsername'
            user_agent_api_form = st.text_input("Reddit User Agent", value=reddit_cfg_api_form.get('user_agent', user_agent_default_form), key="api_reddit_user_agent_form_v10")

        with st.expander("Generative AI API Credentials", expanded=False):
            ai_provider_default_api_form = api_config_for_display.get('ai_provider', 'OpenAI')
            try: ai_provider_idx_api_keys_form = ["OpenAI", "Gemini"].index(ai_provider_default_api_form)
            except ValueError: ai_provider_idx_api_keys_form = 0
            ai_provider_select_api_form = st.selectbox("Select AI Provider", ["OpenAI", "Gemini"], index=ai_provider_idx_api_keys_form, key="api_ai_provider_select_form_v10")
            ai_key_val_api_form = api_config_for_display.get(f'{ai_provider_select_api_form.lower()}_api', {}).get('api_key', '')
            ai_key_input_api_form = st.text_input(f"{ai_provider_select_api_form} API Key", type="password", value=ai_key_val_api_form, key=f"api_ai_key_input_form_v10_{ai_provider_select_api_form}")

        if st.button("Save API Keys", key="api_save_keys_button_form_v10"):
            r_keys_form = {"client_id": client_id_api_form, "client_secret": client_secret_api_form, "user_agent": user_agent_api_form}
            gai_keys_form = {"api_key": ai_key_input_api_form}
            if auth.validate_api_keys(r_keys_form, "Reddit") and auth.validate_api_keys(gai_keys_form, ai_provider_select_api_form):
                if auth.store_api_keys(r_keys_form, gai_keys_form, ai_provider_select_api_form): 
                    st.session_state.project_config = utils.load_project_config(st.session_state.project_path, st.session_state.current_project_name)
            else: ui_helpers.show_warning_message("API key validation failed. Keys not saved.")
    else:
        st.info("Create or open a project to manage its API keys.")


# --- Tab 2: Data Management & View Creation ---
with tab2:
    # ... (Tab 2 code remains the same as the previous full version, ensure widget keys are unique) ...
    st.header("Data Management & View Creation")
    if not st.session_state.current_project_name:
        st.warning("Please create or open a project in Tab 1 first.")
    else:
        st.subheader("Fetch New Reddit Data")
        with st.expander("Show Fetch Options", expanded=True):
            with st.form("reddit_query_form_tab2_v_final_6"): # Unique key
                fetch_subreddit = st.text_input("Subreddit Name", help="e.g., 'learnpython'")
                fetch_query = st.text_input("Search Query/Keywords (optional)")
                fetch_limit = st.number_input("Number of Posts to Fetch", 1, 1000, 25)
                fetch_sort = st.selectbox("Sort Order", ["relevance", "hot", "top", "new"], index=0 if fetch_query else 1)
                fetch_time_filter = st.selectbox("Time Filter (for search/top)", ["all", "year", "month", "week", "day", "hour"])
                submitted_fetch = st.form_submit_button("Fetch & Save Data")

            if submitted_fetch:
                if fetch_subreddit:
                    if not st.session_state.project_config.get('reddit_api') or not st.session_state.project_config.get('reddit_api', {}).get('client_id'):
                        ui_helpers.show_error_message("Reddit API keys not configured in Tab 1.")
                    else:
                        with ui_helpers.show_spinner("Fetching data..."):
                            fetched_df = reddit_api.fetch_reddit_data(fetch_subreddit, fetch_query, fetch_limit, sort=fetch_sort, time_filter=fetch_time_filter)
                        if fetched_df is not None and not fetched_df.empty:
                            current_ts_fetch = datetime.now()
                            fetch_params_meta = {"subreddit": fetch_subreddit, "query": fetch_query, "limit": fetch_limit, "sort": fetch_sort, "time_filter": fetch_time_filter, "timestamp": current_ts_fetch.isoformat()}
                            saved_filepath_fetch = data_management.save_downloaded_reddit_data(fetched_df, fetch_subreddit, fetch_query, fetch_params_meta, current_ts_fetch)
                            if saved_filepath_fetch:
                                ui_helpers.show_success_message(f"Data saved to {os.path.basename(saved_filepath_fetch)}")
                                st.session_state.selected_download_files_info = {} 
                                st.rerun()
                            else: ui_helpers.show_error_message("Failed to save downloaded data.")
                        elif fetched_df is not None: ui_helpers.show_warning_message("No data returned for your query.")
                else: ui_helpers.show_error_message("Subreddit name is required.")
        
        st.divider()
        st.subheader("Manage & Combine Downloaded Datasets")
        downloaded_files_metadata_list = data_management.list_downloaded_files_metadata()

        if not downloaded_files_metadata_list:
            st.info("No Reddit data downloaded for this project yet.")
        else:
            st.markdown("**Available Downloaded Datasets:** (Select to combine)")
            display_meta_for_editor = []
            for meta_item in downloaded_files_metadata_list:
                file_key = meta_item["filename"]
                if file_key not in st.session_state.selected_download_files_info:
                    st.session_state.selected_download_files_info[file_key] = {"selected": False, "filepath": meta_item["filepath"], "metadata": meta_item}
                display_meta_for_editor.append({
                    "Select": st.session_state.selected_download_files_info[file_key].get("selected", False),
                    "File Name": meta_item["filename"], "Subreddit": meta_item.get("subreddit", "N/A"),
                    "Query": meta_item.get("query_used", "N/A"), "Downloaded": meta_item.get("download_timestamp_str", "N/A")
                })
            
            meta_df_for_editor = pd.DataFrame(display_meta_for_editor)
            edited_meta_df_downloads = st.data_editor(
                meta_df_for_editor, disabled=[col for col in meta_df_for_editor.columns if col != "Select"],
                key="downloaded_files_editor_key_v_final_6", hide_index=True, height=min(300, (len(meta_df_for_editor) + 1) * 35 + 3) 
            )

            if not meta_df_for_editor.equals(edited_meta_df_downloads):
                for idx, editor_row in edited_meta_df_downloads.iterrows():
                    filename_editor = editor_row["File Name"]
                    if filename_editor in st.session_state.selected_download_files_info:
                        st.session_state.selected_download_files_info[filename_editor]["selected"] = editor_row["Select"]
                load_and_combine_selected_downloads_for_view()
                st.rerun()

            st.markdown("**Combined Data from Selected Files for View Creation:**")
            combined_df_for_view = st.session_state.get('combined_data_for_view_creation', pd.DataFrame())

            if combined_df_for_view is not None and not combined_df_for_view.empty:
                search_col_view, redact_col_view, redact_btn_view = st.columns([2,2,1])
                with search_col_view:
                    search_key_view = "search_combined_data_tab2_view_v_final_6" 
                    search_term_view = st.text_input("Search Combined Data:", value=st.session_state.search_term_combined_data_tab2, key=search_key_view, on_change=lambda: setattr(st.session_state, 'search_term_combined_data_tab2', st.session_state[search_key_view]))
                    st.session_state.search_term_combined_data_tab2 = search_term_view
                
                df_display_for_view_creation = combined_df_for_view.copy() 
                if search_term_view:
                    df_display_for_view_creation = df_display_for_view_creation[df_display_for_view_creation.astype(str).apply(lambda r: r.str.contains(search_term_view, case=False, na=False).any(), axis=1)]

                with redact_col_view:
                    text_cols_redact_view_creation = [col for col in df_display_for_view_creation.columns if df_display_for_view_creation[col].dtype == 'object']
                    col_to_redact_view_creation = st.selectbox("Column to Redact:", text_cols_redact_view_creation, index=text_cols_redact_view_creation.index('text') if 'text' in text_cols_redact_view_creation else 0, key="redact_col_select_tab2_view_v_final_6") if text_cols_redact_view_creation else None
                
                with redact_btn_view:
                    st.write("") 
                    if col_to_redact_view_creation and st.button("Redact Column (In Memory)", key="redact_button_tab2_view_v_final_6"):
                        st.session_state.redact_confirm_tab2_combined = True
                
                if st.session_state.get("redact_confirm_tab2_combined", False) and col_to_redact_view_creation:
                    def perform_redaction_combined_view_final_6():
                        count = data_management.redact_text_column_in_place(st.session_state.combined_data_for_view_creation, col_to_redact_view_creation)
                        msg = f"Redaction complete. {count} items processed in combined data (in memory)." if count > 0 else "No items identified for redaction."
                        ui_helpers.show_success_message(msg)
                        st.session_state.redact_confirm_tab2_combined = False
                        st.rerun() 
                    
                    st.warning(f"Are you sure you want to redact column '{col_to_redact_view_creation}' in the current Combined Data? This directly modifies the data in memory before view creation.")
                    confirm_cols_final_6 = st.columns(2)
                    if confirm_cols_final_6[0].button("Confirm Redaction", key="confirm_redact_combined_view_btn_final_6"):
                        perform_redaction_combined_view_final_6()
                    if confirm_cols_final_6[1].button("Cancel Redaction", key="cancel_redact_combined_view_btn_final_6"):
                        st.session_state.redact_confirm_tab2_combined = False
                        st.rerun()

                if not df_display_for_view_creation.empty: 
                    st.dataframe(df_display_for_view_creation, height=300)
                    st.divider()
                    view_name_create_input_final = st.text_input("Enter Name for New View:", key="view_name_create_input_tab2_form_final_6")
                    if st.button("Create View from Current Data", key="create_view_button_tab2_form_final_6"):
                        if not view_name_create_input_final: ui_helpers.show_error_message("View name cannot be empty.")
                        else:
                            source_files_info_view_final = [data_info["metadata"]["filename"] for fn, data_info in st.session_state.selected_download_files_info.items() if data_info.get("selected", False)]
                            if data_management.save_project_view(df_display_for_view_creation, view_name_create_input_final, source_filenames_info=source_files_info_view_final):
                                ui_helpers.show_success_message(f"View '{view_name_create_input_final}' created. Proceed to Tab 3 for coding.")
                elif search_term_view: st.info("No data matches your search in the combined dataset.")
                else: st.info("Combined data is empty.")
            elif any(data_info.get("selected", False) for data_info in st.session_state.selected_download_files_info.values()):
                st.info("Loading selected data...")
            else:
                st.info("Select downloaded datasets above to combine them for viewing and view creation.")

# Tab 3: AI Coding on Views
with tab3:
    st.header("AI Coding on Views")
    if not st.session_state.current_project_name:
        st.warning("Please create or open a project in Tab 1 first.")
    else:
        st.subheader("Select Project View(s) for Coding")
        available_views_meta_tab3 = data_management.list_created_views_metadata()

        if not available_views_meta_tab3:
            st.info("No project views created yet. Go to Tab 2 to create a view from downloaded data.")
        else:
            display_views_for_editor_tab3 = []
            for view_meta_item in available_views_meta_tab3:
                view_key_tab3 = view_meta_item["view_name"] 
                if view_key_tab3 not in st.session_state.selected_project_views_info_tab3:
                    st.session_state.selected_project_views_info_tab3[view_key_tab3] = {"selected": False, "metadata": view_meta_item}
                
                display_views_for_editor_tab3.append({
                    "Select": st.session_state.selected_project_views_info_tab3[view_key_tab3].get("selected", False),
                    "View Name": view_meta_item["view_name"],
                    "Created On": datetime.fromisoformat(view_meta_item.get("creation_timestamp", "")).strftime("%Y-%m-%d %H:%M") if view_meta_item.get("creation_timestamp") else "N/A",
                    "Source Files": ", ".join(view_meta_item.get("source_files_info", [])) if isinstance(view_meta_item.get("source_files_info"), list) else str(view_meta_item.get("source_files_info", "N/A"))
                })
            
            views_df_for_editor_tab3 = pd.DataFrame(display_views_for_editor_tab3)
            edited_views_df_for_selection_tab3 = st.data_editor(
                views_df_for_editor_tab3,
                disabled=[col for col in views_df_for_editor_tab3.columns if col != "Select"],
                key="views_selector_editor_tab3_v_final_6", 
                hide_index=True,
                height=min(300, (len(views_df_for_editor_tab3) + 1) * 35 + 3)
            )

            if not views_df_for_editor_tab3.equals(edited_views_df_for_selection_tab3):
                for idx_view_editor, editor_row_view_selected in edited_views_df_for_selection_tab3.iterrows():
                    view_name_from_editor = editor_row_view_selected["View Name"]
                    if view_name_from_editor in st.session_state.selected_project_views_info_tab3:
                        st.session_state.selected_project_views_info_tab3[view_name_from_editor]["selected"] = editor_row_view_selected["Select"]
                load_and_combine_selected_views_for_coding()
                st.rerun()

            st.divider()
            st.subheader("Data for Coding (from selected View(s))")
            df_for_coding_display_tab3 = st.session_state.get('data_for_coding_tab3', pd.DataFrame())

            if df_for_coding_display_tab3.empty:
                st.info("Select one or more views from the table above to load data for coding.")
            else:
                search_coding_data_tab3_input_key = "search_coding_data_tab3_input_v_final_6"
                search_coding_data_tab3_val = st.text_input("Search Coding Data:", 
                                                            value=st.session_state.search_term_coding_data_tab3,
                                                            key=search_coding_data_tab3_input_key,
                                                            on_change=lambda: setattr(st.session_state, 'search_term_coding_data_tab3', st.session_state[search_coding_data_tab3_input_key]))
                st.session_state.search_term_coding_data_tab3 = search_coding_data_tab3_val

                df_display_for_coding_editor = df_for_coding_display_tab3.copy()
                if search_coding_data_tab3_val:
                     df_display_for_coding_editor = df_display_for_coding_editor[df_display_for_coding_editor.astype(str).apply(lambda r: r.str.contains(search_coding_data_tab3_val, case=False, na=False).any(), axis=1)]
                
                st.caption("Edit the 'Codes' column directly (comma-separated). AI will append to existing codes.")
                
                edited_coding_df_from_editor_final = st.data_editor(
                    df_display_for_coding_editor, 
                    column_config={ "Codes": st.column_config.TextColumn("Codes", help="Enter codes separated by commas") },
                    key="coding_data_editor_tab3_v_final_6", 
                    num_rows="dynamic", height=400, use_container_width=True
                )
                
                if not df_display_for_coding_editor.equals(edited_coding_df_from_editor_final):
                    if not search_coding_data_tab3_val: 
                        st.session_state.data_for_coding_tab3 = edited_coding_df_from_editor_final.copy()
                    else: 
                        if 'unique_app_id' in edited_coding_df_from_editor_final.columns and 'unique_app_id' in st.session_state.data_for_coding_tab3.columns:
                            update_dict = edited_coding_df_from_editor_final.set_index('unique_app_id')['Codes'].to_dict()
                            main_df_codes = st.session_state.data_for_coding_tab3.set_index('unique_app_id')
                            for uid, new_codes in update_dict.items():
                                if uid in main_df_codes.index:
                                    main_df_codes.loc[uid, 'Codes'] = new_codes
                            st.session_state.data_for_coding_tab3 = main_df_codes.reset_index()
                        else:
                            st.warning("Cannot reliably merge edits from searched data without 'unique_app_id'. Clear search to edit.")
                
                st.subheader("AI Code Generation")
                ai_text_col_options_batch_final = [col for col in df_for_coding_display_tab3.columns if df_for_coding_display_tab3[col].dtype == 'object' and col not in ['Source View', 'Codes', 'unique_app_id', 'id']]
                if not ai_text_col_options_batch_final:
                    st.warning("No suitable text columns found for AI coding.")
                else:
                    default_text_col_idx_batch_final = ai_text_col_options_batch_final.index('text') if 'text' in ai_text_col_options_batch_final else 0
                    st.session_state.col_for_coding_tab3 = st.selectbox( 
                        "Select text column for AI input:", ai_text_col_options_batch_final, 
                        index=default_text_col_idx_batch_final, key="ai_text_col_select_tab3_batch_v_final_6"
                    )
                    
                    id_col_for_ai_batch = st.session_state.id_col_for_coding_tab3 # Should be unique_app_id
                    if id_col_for_ai_batch not in df_for_coding_display_tab3.columns:
                         st.error(f"ID column '{id_col_for_ai_batch}' not found in data. Cannot proceed with batch AI coding.")
                         id_col_for_ai_batch = None # Prevent button from running

                    if id_col_for_ai_batch:
                        st.caption(f"Using column '`{id_col_for_ai_batch}`' as the unique identifier for matching AI results.")

                        ai_provider_batch_final = st.session_state.project_config.get('ai_provider', "OpenAI")
                        ai_model_batch_final = None
                        if ai_provider_batch_final == "OpenAI": ai_model_batch_final = st.selectbox("Select OpenAI Model:", ["gpt-3.5-turbo", "gpt-4", "gpt-4o"], key="openai_model_coding_tab3_batch_v_final_6")
                        elif ai_provider_batch_final == "Gemini": ai_model_batch_final = st.selectbox("Select Gemini Model:", ["gemini-2.0-flash", "gemini-1.5-flash"], key="gemini_model_coding_tab3_batch_v_final_6")
                        
                        batch_prompt_template_corrected_final = st.text_area(
                            "AI Batch Coding Prompt Template:", 
                            value="""Perform qualitative thematic analysis on the provided JSON array of Reddit data.
Each item in the input array has a "unique_app_id" and a text field named "{text_column_name}".
For each item, use the text in its "{text_column_name}" field to generate up to 5 concise, meaningful thematic codes.

Return **only** a valid JSON array where each element is an object containing two fields:
1.  "unique_app_id": the original unique identifier from the input item.
2.  "Codes": a single string of comma-separated thematic codes generated from the text.

Example of a single element in the **output** JSON array you should return:
{
  "unique_app_id": "some_original_id_123",
  "Codes": "theme alpha, topic beta, user concern gamma"
}

Do not include any introductory text, explanations, markdown formatting, or any characters outside the main JSON array structure.
The input data batch you need to process is below:
{json_data_batch}""", 
                            key="ai_batch_prompt_template_tab3_v_final_6", height=350,
                            help="Use {text_column_name} for the selected text column name placeholder and {json_data_batch} for the data placeholder."
                        )
                        st.session_state.ai_batch_size_tab3 = st.number_input("Batch Size for AI Processing:", min_value=1, max_value=50, value=st.session_state.ai_batch_size_tab3, step=1, key="batch_size_tab3_input_v_final_6")

                        if st.button("Generate Codes with AI (Batch Processing)", key="generate_ai_codes_batch_tab3_btn_v_final_6", disabled=(not id_col_for_ai_batch)):
                            text_col_for_ai_run = st.session_state.col_for_coding_tab3
                            if not text_col_for_ai_run or not ai_model_batch_final or \
                               '{json_data_batch}' not in batch_prompt_template_corrected_final or \
                               '{text_column_name}' not in batch_prompt_template_corrected_final:
                                ui_helpers.show_error_message("Select text column, AI model, and ensure prompt template placeholders are correct.")
                            elif st.session_state.data_for_coding_tab3.empty:
                                ui_helpers.show_error_message("No data loaded in the coding table.")
                            else:
                                df_to_code_run = st.session_state.data_for_coding_tab3.copy()
                                if text_col_for_ai_run not in df_to_code_run.columns or id_col_for_ai_batch not in df_to_code_run.columns:
                                    ui_helpers.show_error_message(f"Selected text column '{text_col_for_ai_run}' or ID column '{id_col_for_ai_batch}' not found.")
                                else:
                                    num_batches_run = (len(df_to_code_run) - 1) // st.session_state.ai_batch_size_tab3 + 1
                                    progress_bar_ai_run = st.progress(0, text=f"Starting AI batch coding (0/{num_batches_run} batches)...")
                                    all_ai_processed_results_run = []
                                    final_prompt_template_run = batch_prompt_template_corrected_final.replace("{text_column_name}", text_col_for_ai_run)

                                    for i_batch in range(num_batches_run):
                                        batch_start_idx_run = i_batch * st.session_state.ai_batch_size_tab3
                                        batch_end_idx_run = batch_start_idx_run + st.session_state.ai_batch_size_tab3
                                        current_batch_df_run = df_to_code_run.iloc[batch_start_idx_run:batch_end_idx_run]
                                        progress_bar_ai_run.progress((i_batch + 1) / num_batches_run, text=f"Processing batch {i_batch+1}/{num_batches_run}...")
                                        batch_ai_output_run = ai_services.generate_codes_for_batch_with_ai(
                                            current_batch_df_run, ai_provider_batch_final, final_prompt_template_run,
                                            text_column_name=text_col_for_ai_run, id_column_name=id_col_for_ai_batch, model_name=ai_model_batch_final
                                        )
                                        if batch_ai_output_run: all_ai_processed_results_run.extend(batch_ai_output_run)
                                        else: ui_helpers.show_error_message(f"Batch {i_batch+1} failed. Stopping."); break 
                                    progress_bar_ai_run.empty()

                                    if all_ai_processed_results_run:
                                        ai_results_df_run = pd.DataFrame(all_ai_processed_results_run)
                                        # Ensure the ID column name used by AI matches the one in our df_for_coding
                                        ai_results_df_run.rename(columns={'unique_app_id': id_col_for_ai_batch, 'Codes': 'AI_Generated_Codes'}, inplace=True)
                                        
                                        df_to_update_session = st.session_state.data_for_coding_tab3.copy()
                                        df_to_update_session[id_col_for_ai_batch] = df_to_update_session[id_col_for_ai_batch].astype(str)
                                        ai_results_df_run[id_col_for_ai_batch] = ai_results_df_run[id_col_for_ai_batch].astype(str)

                                        # Merge AI results
                                        merged_df_for_codes = pd.merge(
                                            df_to_update_session,
                                            ai_results_df_run[[id_col_for_ai_batch, 'AI_Generated_Codes', 'error']],
                                            on=id_col_for_ai_batch, how='left'
                                        )
                                        newly_coded_rows_run = 0
                                        for idx_merge, row_merge in merged_df_for_codes.iterrows():
                                            if pd.notna(row_merge['AI_Generated_Codes']) and str(row_merge['AI_Generated_Codes']).strip():
                                                existing_codes_list_merge = robust_comma_string_to_list(row_merge['Codes'])
                                                ai_new_codes_list_merge = robust_comma_string_to_list(row_merge['AI_Generated_Codes'])
                                                combined_codes_merge = list(dict.fromkeys(existing_codes_list_merge + ai_new_codes_list_merge))
                                                merged_df_for_codes.loc[idx_merge, 'Codes'] = robust_list_to_comma_string(combined_codes_merge)
                                                newly_coded_rows_run +=1
                                            if pd.notna(row_merge['error']): logger.warning(f"AI error for {row_merge[id_col_for_ai_batch]}: {row_merge['error']}")
                                        st.session_state.data_for_coding_tab3 = merged_df_for_codes.drop(columns=['AI_Generated_Codes', 'error'], errors='ignore')
                                        ui_helpers.show_success_message(f"AI batch coding complete. {newly_coded_rows_run} items potentially updated.")
                                        st.rerun()
                                    else: ui_helpers.show_error_message("AI batch coding returned no results or failed.")
                
                st.divider()
                # ... (Save Coded Data button logic - same as previous full app.py) ...
                if st.button("Save Coded Data to View(s)", key="save_coded_data_tab3_btn_v_final_6"):
                    if st.session_state.data_for_coding_tab3.empty:
                        ui_helpers.show_warning_message("No data with codes to save.")
                    else:
                        df_to_save_with_codes_final = st.session_state.data_for_coding_tab3.copy()
                        selected_view_paths_final = st.session_state.get('currently_selected_view_filepaths_for_saving_tab3', [])
                        if len(selected_view_paths_final) == 1:
                            view_csv_path_to_save_single_final = selected_view_paths_final[0]
                            if data_management.save_coded_data_to_view(df_to_save_with_codes_final.drop(columns=['Source View'], errors='ignore'), view_csv_path_to_save_single_final):
                                ui_helpers.show_success_message(f"Codes saved to view: '{os.path.basename(view_csv_path_to_save_single_final)}'.")
                            else:
                                ui_helpers.show_error_message(f"Failed to save codes to view: '{os.path.basename(view_csv_path_to_save_single_final)}'.")
                        elif len(selected_view_paths_final) > 1:
                            st.session_state.prompt_save_combined_as_new_view_tab3 = True
                        else: 
                            st.session_state.prompt_save_combined_as_new_view_tab3 = True

                if st.session_state.get("prompt_save_combined_as_new_view_tab3", False):
                    st.info("Save the current coded data (possibly from combined views) as a new project view.")
                    new_coded_view_name_final = st.text_input("Enter name for this new coded view:", key="new_coded_view_name_input_v_final_6")
                    if st.button("Save as New Coded View", key="save_as_new_coded_view_btn_v_final_6"):
                        if not new_coded_view_name_final:
                            ui_helpers.show_error_message("New view name cannot be empty.")
                        else:
                            source_info_for_new_coded_view_final = [
                                info["metadata"]["view_name"] for view_name_key, info in st.session_state.selected_project_views_info_tab3.items() if info.get("selected", False)
                            ]
                            if not source_info_for_new_coded_view_final: source_info_for_new_coded_view_final = ["Combined/Edited Data"]
                            
                            df_to_save_new_view_final = st.session_state.data_for_coding_tab3.copy()
                            if data_management.save_project_view(df_to_save_new_view_final.drop(columns=['Source View'], errors='ignore'), new_coded_view_name_final, source_filenames_info=source_info_for_new_coded_view_final):
                                ui_helpers.show_success_message(f"Combined coded data saved as new view: '{new_coded_view_name_final}'.")
                                st.session_state.prompt_save_combined_as_new_view_tab3 = False
                                st.session_state.selected_project_views_info_tab3 = {}
                                st.session_state.data_for_coding_tab3 = pd.DataFrame()
                                st.rerun()

with tab4: # Analysis & Visualization
    st.header("Analysis & Visualization")
    if not st.session_state.current_project_name:
        st.warning("Please create or open a project in Tab 1 first.")
    else:
        st.info("Tab 4: Analyze combined views with filters and visualizations. (Implementation Pending)")

# --- Sidebar ---
# ... (Sidebar code - same as previous full version) ...
st.sidebar.info(
    """
    **Qualitative Thematic Analysis Tool**
    **Workflow:**
    1. **Tab 1:** Create or Open a Project. Setup storage & API keys.
    2. **Tab 2:** Fetch & manage Reddit downloads. Combine data and create 'Views'.
    3. **Tab 3:** Select View(s) for AI coding and manual refinement.
    4. **Tab 4:** Analyze and visualize coded data from views.
    **Notes:** User specifies directory for project config files.
    """
)
st.sidebar.markdown("---")
if st.session_state.current_project_name and st.session_state.project_path:
    st.sidebar.success(f"Active Project: {st.session_state.current_project_name}")
    project_sidebar_path_display_final = st.session_state.project_path
    active_storage_sidebar = st.session_state.project_config.get('storage_type', 'N/A')
    st.sidebar.caption(f"Storage Type: {active_storage_sidebar}")
    st.sidebar.caption(f"Config/Data Dir: `{os.path.normpath(project_sidebar_path_display_final)}`")
else:
    st.sidebar.warning("No active project.")

if st.session_state.current_project_name and not st.session_state.project_config.get('project_config_file_directory'):
    logger.warning("Active project but 'project_config_file_directory' missing. Attempting recovery.")
    if st.session_state.project_path and os.path.exists(st.session_state.project_path):
        loaded_cfg_recovery = utils.load_project_config(st.session_state.project_path, st.session_state.current_project_name)
        if loaded_cfg_recovery:
            st.session_state.project_config = loaded_cfg_recovery
            st.session_state.project_path = loaded_cfg_recovery.get('project_config_file_directory', st.session_state.project_path)
            initialize_project_data_states()
        else: logger.error(f"Recovery failed for {st.session_state.current_project_name}")
    else: logger.warning("Cannot attempt recovery: project_path invalid.")