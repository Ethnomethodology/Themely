# app.py
import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os # Ensure os is imported

# Import modules from the 'modules' directory
from modules import auth, reddit_api, ai_services, data_management, ui_helpers, utils

# --- Page Configuration (Set this at the very beginning) ---
st.set_page_config(layout="wide", page_title="Qualitative Thematic Analysis Tool")

logger = utils.setup_logger("app")

# --- Initialize Session State ---
if 'current_project_name' not in st.session_state:
    st.session_state.current_project_name = None
if 'project_config' not in st.session_state: # Holds API keys, paths etc. for current project
    st.session_state.project_config = {}
if 'project_path' not in st.session_state: # Path to the project's FOLDER (where config & data files are stored)
    st.session_state.project_path = None
if 'raw_data' not in st.session_state: # DataFrame for Reddit data
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state: # DataFrame for data after redaction, coding
    st.session_state.processed_data = None
if 'current_view_data' not in st.session_state: # For filtered views
    st.session_state.current_view_data = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Project Setup & API Config" # Default active tab
if 'col_for_coding' not in st.session_state: # To remember the column selected for coding
    st.session_state.col_for_coding = None


# --- Helper function to load project data ---
def load_project_if_selected():
    if st.session_state.current_project_name and st.session_state.project_path:
        # Load project config first if not already loaded or to refresh
        # The project_path in session state is the folder where the project's config is stored.
        # The project_name is used to construct the config filename within that folder.
        loaded_cfg = utils.load_project_config(st.session_state.project_path, st.session_state.current_project_name)
        if loaded_cfg:
            st.session_state.project_config = loaded_cfg
        else:
            # This might happen if config file was deleted manually, or if it's a new project setup step
            logger.warning(f"Could not load project config for {st.session_state.current_project_name} from {st.session_state.project_path}. Might be new.")


        # Try to load existing data for this project
        # data_management functions will use st.session_state.project_path
        if st.session_state.raw_data is None:
             st.session_state.raw_data = data_management.load_data_from_project("reddit_data.csv")
        if st.session_state.processed_data is None:
             st.session_state.processed_data = data_management.load_data_from_project("processed_data.csv")
             if st.session_state.processed_data is not None:
                 # Ensure 'ai_codes' column is list, not string, after CSV load
                 if 'ai_codes' in st.session_state.processed_data.columns:
                     try:
                         # Convert string representation of list back to list
                         # Also handle cases where it might already be a list (e.g., if not from CSV)
                         # or if it's NaN
                         def robust_eval_list(x):
                             if isinstance(x, list):
                                 return x
                             if pd.isna(x):
                                 return [] # Or None, depending on desired handling for missing codes
                             if isinstance(x, str):
                                 try:
                                     evaluated = eval(x)
                                     return evaluated if isinstance(evaluated, list) else []
                                 except:
                                     # Fallback for simple comma separated or other non-list strings
                                     # This might need adjustment based on how strings are actually stored if not list-like
                                     return [s.strip() for s in x.split(',') if s.strip()] if x.strip() else []
                             return [] # Default for other types

                         st.session_state.processed_data['ai_codes'] = st.session_state.processed_data['ai_codes'].apply(robust_eval_list)
                     except Exception as e:
                         logger.warning(f"Could not auto-convert 'ai_codes' column to list: {e}. Manual check might be needed.")


# --- Main App ---
st.title("Qualitative Thematic Analysis for Reddit Data")

# --- Tabbed Interface ---
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Project Setup & API Config",
    "2. Data Retrieval, Viewing, Editing & Redaction",
    "3. AI Coding, Filtering & View Management",
    "4. Clustering, Theme Analysis & Visualization"
])


# ========================== TAB 1: Project Setup & API Config ==========================
with tab1:
    st.header("Project Setup")

    # Project Name and Path
    project_name_input = st.text_input("Project Name", key="project_name_input", value=st.session_state.get("current_project_name", ""))

    # Storage Selection
    storage_options = ["Local", "Google Drive", "Dropbox", "OneDrive"]
    # Get current storage type from project_config if project is loaded
    current_storage_type = st.session_state.project_config.get('storage_type', "Local")
    try:
        current_storage_index = storage_options.index(current_storage_type)
    except ValueError:
        current_storage_index = 0 # Default to Local if stored type is somehow invalid

    selected_storage = st.selectbox(
        "Select Storage Type",
        storage_options,
        index=current_storage_index,
        key="storage_type_selector",
        help="Cloud options like Google Drive & Dropbox are conceptual placeholders. OneDrive has a functional device flow authentication."
    )

    user_chosen_local_base_path = "" # Initialize
    # Path for storing config files for cloud projects (always local for this app's design)
    cloud_project_config_base_path = "data/project_configs_for_cloud_storage/"


    if selected_storage == "Local":
        st.markdown("""
        **Local Storage Setup:**
        Please provide the **absolute path** to a folder on your computer where you want to store your projects.
        A new subfolder for *this specific project* (e.g., `Your_Project_Name_ID`) will be created inside the path you provide.
        Example: If you enter `/Users/yourname/Documents/MyResearchProjects`, this project will be saved in
        `/Users/yourname/Documents/MyResearchProjects/Your_Project_Name_ID/`.
        """)
        # Retrieve the last used local base path for convenience if available
        default_local_base = st.session_state.project_config.get('user_local_base_path_for_projects', os.path.expanduser("~")) # Default to user's home directory
        user_chosen_local_base_path = st.text_input(
            "Enter absolute path to your desired projects folder:",
            value=default_local_base,
            key="user_chosen_local_base_path_input",
            help="E.g., C:\\Users\\YourName\\Documents\\Projects or /Users/yourname/Projects"
        )
    else: # Cloud Storage
        st.info(f"""
            For **{selected_storage}**:
            - **OneDrive**: Uses Device Flow for authentication. Ensure you have set `ONEDRIVE_CLIENT_ID` in `modules/auth.py`.
            - **Google Drive/Dropbox**: These are currently conceptual placeholders and would require full OAuth2 client setup.

            *Project configuration files for cloud-based projects will be stored locally in a dedicated folder: `{os.path.abspath(cloud_project_config_base_path)}`.*
            *The actual project data (Reddit data, etc.) would reside in your chosen cloud storage, managed by the respective API after authentication.*
            """)
        # For cloud storage, the `setup_project_storage` will use `cloud_project_config_base_path`
        # to determine where the local YAML config file for that cloud project is stored.


    if st.button("Create/Load Project", key="create_load_project_button"):
        if project_name_input:
            path_for_setup = ""
            if selected_storage == "Local":
                if not user_chosen_local_base_path:
                    ui_helpers.show_error_message("For local storage, the absolute path to your projects folder cannot be empty.")
                    st.stop()
                if not os.path.isabs(user_chosen_local_base_path):
                    ui_helpers.show_error_message("Please provide an absolute path for your local projects folder (e.g., starting with C:\\ or /).")
                    st.stop()
                path_for_setup = user_chosen_local_base_path
            else: # Cloud storage
                path_for_setup = cloud_project_config_base_path # This is where the local config for the cloud project will live

            if auth.setup_project_storage(project_name_input, selected_storage, designated_path=path_for_setup):
                st.session_state.current_project_name = project_name_input
                if st.session_state.project_path: # project_path is set by setup_project_storage
                     st.session_state.project_config = utils.load_project_config(st.session_state.project_path, project_name_input)
                     if selected_storage == "Local": # Store the chosen base path for future convenience
                         st.session_state.project_config['user_local_base_path_for_projects'] = user_chosen_local_base_path
                         utils.save_project_config(st.session_state.project_path, project_name_input, st.session_state.project_config)

                st.session_state.raw_data = None
                st.session_state.processed_data = None
                st.session_state.current_view_data = None
                load_project_if_selected()
                st.rerun()
            else:
                ui_helpers.show_error_message("Failed to setup or load project. Check messages above and console for errors.")
        else:
            ui_helpers.show_error_message("Project name cannot be empty.")

    if st.session_state.current_project_name and st.session_state.project_path:
        project_display_path = st.session_state.project_path
        if st.session_state.project_config.get('storage_type') != "Local":
            project_display_path = f"{st.session_state.project_config.get('storage_type')} (Config at: {st.session_state.project_path})"

        st.success(f"Current Project: **{st.session_state.current_project_name}**")
        st.caption(f"Storage Type: {st.session_state.project_config.get('storage_type', 'N/A')}")
        st.caption(f"Project Location/Config: '{project_display_path}'")


        if st.session_state.project_config.get('storage_type') == "OneDrive":
            if st.button("Re-authenticate OneDrive", key="reauth_onedrive_button_tab1"):
                st.session_state.force_onedrive_reauth = True
                project_name_current = st.session_state.current_project_name
                # The base path for local storage of OneDrive project config files
                base_path_for_onedrive_config = cloud_project_config_base_path

                if auth.setup_project_storage(project_name_current, "OneDrive", designated_path=base_path_for_onedrive_config):
                     if st.session_state.project_path:
                        st.session_state.project_config = utils.load_project_config(st.session_state.project_path, project_name_current)
                     ui_helpers.show_success_message("OneDrive re-authentication process completed.")
                else:
                    ui_helpers.show_error_message("OneDrive re-authentication process failed or was cancelled.")
                if 'force_onedrive_reauth' in st.session_state:
                    del st.session_state['force_onedrive_reauth']
                st.rerun()
    else:
        st.warning("No project loaded or created yet.")

    st.divider()
    st.header("API Key Management")

    if st.session_state.current_project_name and st.session_state.project_path:
        current_config_for_api_display = utils.load_project_config(st.session_state.project_path, st.session_state.current_project_name)
        if not current_config_for_api_display:
            current_config_for_api_display = st.session_state.project_config

        with st.expander("Reddit API Credentials", expanded=False):
            reddit_api_cfg = current_config_for_api_display.get('reddit_api', {})
            reddit_client_id = st.text_input("Reddit Client ID", type="password", value=reddit_api_cfg.get('client_id', ''), help="Your Reddit application's client ID.")
            reddit_client_secret = st.text_input("Reddit Client Secret", type="password", value=reddit_api_cfg.get('client_secret', ''), help="Your Reddit application's client secret.")
            default_user_agent = f'streamlit_thematic_app_v1 by /u/your_username_{utils.generate_project_id(st.session_state.current_project_name)}'
            reddit_user_agent = st.text_input("Reddit User Agent", value=reddit_api_cfg.get('user_agent', default_user_agent), help="A unique user agent string.")

        with st.expander("Generative AI API Credentials", expanded=False):
            default_ai_provider = current_config_for_api_display.get('ai_provider', 'OpenAI')
            try:
                ai_provider_idx = ["OpenAI", "Gemini"].index(default_ai_provider)
            except ValueError: ai_provider_idx = 0
            ai_provider = st.selectbox("Select AI Provider", ["OpenAI", "Gemini"], index=ai_provider_idx, key="ai_provider_select_tab1")
            ai_api_key_val = current_config_for_api_display.get(f'{ai_provider.lower()}_api', {}).get('api_key', '')
            ai_api_key = st.text_input(f"{ai_provider} API Key", type="password", value=ai_api_key_val, help=f"Your API key for {ai_provider}.", key=f"ai_api_key_input_{ai_provider}")

        if st.button("Save API Keys", key="save_api_keys_button_tab1"):
            reddit_keys = {"client_id": reddit_client_id, "client_secret": reddit_client_secret, "user_agent": reddit_user_agent}
            ai_keys = {"api_key": ai_api_key}
            valid_reddit = auth.validate_api_keys(reddit_keys, "Reddit")
            valid_ai = auth.validate_api_keys(ai_keys, ai_provider)
            if valid_reddit and valid_ai:
                if auth.store_api_keys(reddit_keys, ai_keys, ai_provider):
                    st.session_state.project_config = utils.load_project_config(st.session_state.project_path, st.session_state.current_project_name)
            else:
                ui_helpers.show_warning_message("One or more API key sets are incomplete or invalid. Keys not saved.")
    else:
        st.info("Please create or load a project to manage API keys.")


# ========================== TAB 2: Data Retrieval, Viewing, Editing & Redaction ==========================
with tab2:
    st.header("Reddit Data Retrieval")
    if not st.session_state.current_project_name or not st.session_state.project_config.get('reddit_api') or not st.session_state.project_config.get('reddit_api', {}).get('client_id'):
        st.warning("Please set up a project and valid Reddit API keys in Tab 1 first.")
    else:
        load_project_if_selected()

        with st.form("reddit_query_form"):
            subreddit = st.text_input("Subreddit Name (e.g., 'python')", help="Enter the name of the subreddit without 'r/'.")
            query = st.text_input("Search Query/Keywords (optional)", help="Keywords to search for within the subreddit. Leave blank to fetch general posts from sort order.")
            limit = st.number_input("Number of Posts/Items to Fetch", min_value=1, max_value=1000, value=50, step=10)
            sort_order = st.selectbox("Sort Order (for search or general fetch if no query)", ["relevance", "hot", "top", "new", "comments"], index=0 if query else 1)
            time_filter = st.selectbox("Time Filter (for search or 'top'/'controversial' sort)", ["all", "year", "month", "week", "day", "hour"], index=0)
            submitted_query = st.form_submit_button("Fetch Data from Reddit")

        if submitted_query:
            if subreddit:
                with ui_helpers.show_spinner("Fetching data... This may take a while."):
                    retrieved_df = reddit_api.fetch_reddit_data(subreddit, query, limit, search_type="posts", time_filter=time_filter, sort=sort_order)
                if retrieved_df is not None and not retrieved_df.empty:
                    st.session_state.raw_data = retrieved_df
                    st.session_state.processed_data = retrieved_df.copy()
                    data_management.save_data_to_project(st.session_state.raw_data, "reddit_data.csv")
                    data_management.save_data_to_project(st.session_state.processed_data, "processed_data.csv")
                elif retrieved_df is not None and retrieved_df.empty and not st.session_state.get('cancel_fetch', False):
                     ui_helpers.show_warning_message("No data returned for your query. Try different parameters.")
            else:
                ui_helpers.show_error_message("Subreddit name is required.")

        if st.button("Cancel Current Data Fetch", key="cancel_fetch_button_main_ui"):
            st.session_state.cancel_fetch = True
            ui_helpers.show_warning_message("Cancellation requested. Will attempt to stop after the current item or next check.")

        st.divider()
        st.header("Data Viewing, Editing, and Redaction")
        active_data_source = st.radio("Select data to view/edit:",
                                      ["Raw Data (Original Fetch)", "Processed Data (After Redaction/Coding)"],
                                      index=1 if st.session_state.processed_data is not None else 0,
                                      horizontal=True, key="data_source_selector")
        current_df_to_display = None
        data_key_for_saving = None
        if active_data_source == "Raw Data (Original Fetch)":
            current_df_to_display = st.session_state.raw_data
            data_key_for_saving = "raw_data"
            st.info("Displaying Raw Data. Edits here are generally not recommended. Consider editing 'Processed Data'.")
        else:
            current_df_to_display = st.session_state.processed_data
            data_key_for_saving = "processed_data"

        if current_df_to_display is not None and not current_df_to_display.empty:
            st.info(f"Displaying: {active_data_source}. Rows: {len(current_df_to_display)}, Columns: {len(current_df_to_display.columns)}")
            search_term = st.text_input("Search table data (case-insensitive)", key="table_search_main")
            if search_term:
                df_display_filtered = current_df_to_display[current_df_to_display.astype(str).apply(lambda row: row.str.contains(search_term, case=False, na=False).any(), axis=1)]
            else:
                df_display_filtered = current_df_to_display.copy()
            st.subheader("Interactive Data Table")
            edited_df = ui_helpers.display_interactive_table(df_display_filtered, key=f"data_editor_{data_key_for_saving}")
            if edited_df is not None:
                if not search_term:
                    if data_key_for_saving == "processed_data": st.session_state.processed_data = edited_df
                    elif data_key_for_saving == "raw_data": st.session_state.raw_data = edited_df
                else:
                    st.warning("Edits on a filtered view are reflected in this view only. Clear search to edit the full dataset or implement advanced merging.")
            if st.button(f"Save Changes to {active_data_source}", key=f"save_edited_{data_key_for_saving}"):
                if search_term:
                    ui_helpers.show_warning_message("Cannot save directly from a filtered view. Clear search and make changes on the full data.")
                elif data_key_for_saving == "processed_data" and st.session_state.processed_data is not None:
                    if data_management.save_data_to_project(st.session_state.processed_data, "processed_data.csv"):
                        ui_helpers.show_success_message(f"Changes saved to 'processed_data.csv' in project folder.")
                elif data_key_for_saving == "raw_data" and st.session_state.raw_data is not None:
                    if data_management.save_data_to_project(st.session_state.raw_data, "reddit_data.csv"):
                         ui_helpers.show_success_message(f"Changes saved to 'reddit_data.csv' in project folder.")
            st.subheader("Data Redaction (using Microsoft Presidio)")
            if st.session_state.processed_data is not None and not st.session_state.processed_data.empty:
                text_columns = [col for col in st.session_state.processed_data.columns if st.session_state.processed_data[col].dtype == 'object']
                if not text_columns: st.warning("No text columns in Processed Data for redaction.")
                else:
                    default_text_col_idx = text_columns.index('text') if 'text' in text_columns else (text_columns.index('title') if 'title' in text_columns else 0)
                    column_to_redact = st.selectbox("Select text column for redaction (from Processed Data):", text_columns, index=default_text_col_idx, key="col_to_redact_select")
                    if st.button(f"Redact Column: '{column_to_redact}'", key="redact_button_confirm"):
                        st.session_state.show_redact_confirm_dialog = True
                    if st.session_state.get("show_redact_confirm_dialog", False):
                        def perform_redaction_callback():
                            with ui_helpers.show_spinner(f"Redacting '{column_to_redact}'..."):
                                df_redacted, count = data_management.redact_text_column(st.session_state.processed_data, column_to_redact)
                            st.session_state.processed_data = df_redacted
                            data_management.save_data_to_project(st.session_state.processed_data, "processed_data.csv")
                            msg = f"Redaction complete. {count} items processed." if count > 0 else f"Redaction ran for '{column_to_redact}', but no items were identified by Presidio."
                            ui_helpers.show_success_message(f"{msg} New column '{column_to_redact}_redacted' added/updated.")
                            st.session_state.show_redact_confirm_dialog = False
                            st.rerun()
                        ui_helpers.confirm_action_dialog(
                            message=f"Redact column '{column_to_redact}' in 'Processed Data'? This creates/overwrites '{column_to_redact}_redacted'. Original column remains.",
                            action_callback=perform_redaction_callback, key_suffix="redact_main")
            else: st.info("Load/fetch data into 'Processed Data' for redaction.")
        else: st.info("No data loaded. Fetch data or ensure a project with data is active.")

# ========================== TAB 3: AI Coding, Filtering & View Management ==========================
with tab3:
    st.header("AI-Assisted Coding")
    if st.session_state.processed_data is None or st.session_state.processed_data.empty:
        st.warning("Ensure 'Processed Data' is available (Tab 2) before coding.")
    elif not st.session_state.project_config.get('ai_provider') or not st.session_state.project_config.get(f"{st.session_state.project_config.get('ai_provider', '').lower()}_api", {}).get('api_key'):
        st.warning("Please configure AI Provider and API Key in Tab 1.")
    else:
        load_project_if_selected()
        available_text_cols = [col for col in st.session_state.processed_data.columns if st.session_state.processed_data[col].dtype == 'object']
        if not available_text_cols: st.error("No text columns in Processed Data for AI coding."); st.stop()
        last_selected_col = st.session_state.get('col_for_coding')
        default_col_for_coding = last_selected_col if last_selected_col and last_selected_col in available_text_cols else \
                                 ('text_redacted' if 'text_redacted' in available_text_cols else \
                                  ('text' if 'text' in available_text_cols else \
                                   ('title_redacted' if 'title_redacted' in available_text_cols else \
                                    ('title' if 'title' in available_text_cols else available_text_cols[0]))))
        default_coding_col_idx = available_text_cols.index(default_col_for_coding) if default_col_for_coding in available_text_cols else 0
        col_to_code_selected = st.selectbox("Select text column for AI coding (from Processed Data):", available_text_cols, index=default_coding_col_idx, key="col_for_coding_selector", help="Choose the column for coding. Redacted columns preferred.")
        st.session_state.col_for_coding = col_to_code_selected
        ai_provider = st.session_state.project_config.get('ai_provider', "OpenAI")
        st.info(f"Using AI Provider: **{ai_provider}** (configured in Tab 1).")
        ai_model = None
        if ai_provider == "OpenAI": ai_model = st.selectbox("Select OpenAI Model:", ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"], index=0)
        elif ai_provider == "Gemini": ai_model = st.selectbox("Select Gemini Model:", ["gemini-pro", "gemini-1.0-pro", "gemini-1.5-flash", "gemini-1.5-pro"], index=0)
        else: st.error("AI Provider not properly configured.")
        coding_prompt_template = st.text_area("AI Coding Prompt Template", value="Analyze the following text and generate a comma-separated list of 3-5 concise thematic codes or keywords that capture its main topics or sentiments. Text: {text}", height=100, help="Use '{text}' as a placeholder.")
        if st.button(f"Generate AI Codes for '{st.session_state.col_for_coding}'", key="generate_codes_button_main") and st.session_state.col_for_coding:
            if '{text}' not in coding_prompt_template: ui_helpers.show_error_message("Prompt template must include '{text}' placeholder.")
            else:
                texts_to_code = st.session_state.processed_data[st.session_state.col_for_coding].fillna("").astype(str).tolist()
                with ui_helpers.show_spinner(f"Generating codes with {ai_provider} for {len(texts_to_code)} items..."):
                    ai_coding_results = ai_services.generate_codes_with_ai(texts_to_code, ai_provider, coding_prompt_template, model_name=ai_model)
                if ai_coding_results:
                    codes_for_df = [res['codes'] if res and 'codes' in res and isinstance(res['codes'], list) else [] for res in ai_coding_results]
                    errors_for_df = [res['error'] if res and 'error' in res else None for res in ai_coding_results]
                    st.session_state.processed_data['ai_codes'] = codes_for_df
                    st.session_state.processed_data['ai_coding_errors'] = errors_for_df
                    data_management.save_data_to_project(st.session_state.processed_data, "processed_data.csv")
                    ui_helpers.show_success_message(f"AI coding complete. 'ai_codes' and 'ai_coding_errors' columns added/updated.")
                    num_errors = sum(1 for err in errors_for_df if err is not None)
                    if num_errors > 0: ui_helpers.show_warning_message(f"{num_errors} items encountered errors during coding. Check 'ai_coding_errors' column.")
                    st.rerun()
                else: ui_helpers.show_error_message("AI coding failed or returned no results.")
        if 'ai_codes' in st.session_state.processed_data.columns:
            st.subheader("View/Edit Generated Codes")
            st.markdown("To view and edit codes, go to **Tab 2**, select **Processed Data**, and use the interactive table. Ensure codes are Python lists of strings, e.g., `['code A', 'code B']`. Save changes in Tab 2. Preview below:")
            if st.session_state.col_for_coding and st.session_state.col_for_coding in st.session_state.processed_data.columns:
                st.dataframe(st.session_state.processed_data[[st.session_state.col_for_coding, 'ai_codes']].head())
        st.divider()
        st.header("Filtering, Grouping, and Saving Views")
        if st.session_state.processed_data is not None and 'ai_codes' in st.session_state.processed_data.columns:
            st.subheader("Filter Data by Codes")
            st.session_state.processed_data['ai_codes'] = st.session_state.processed_data['ai_codes'].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else (eval(x) if isinstance(x, str) and x.startswith('[') else [str(x)])))
            all_unique_codes = sorted(list(set(code for sublist in st.session_state.processed_data['ai_codes'].dropna() for code in sublist if code)))
            if not all_unique_codes: st.info("No codes in 'Processed Data' for filtering.")
            else:
                selected_codes_for_filter = st.multiselect("Select codes to filter by (shows data with ANY selected code):", all_unique_codes, key="filter_code_multi_select")
                if selected_codes_for_filter:
                    mask = st.session_state.processed_data['ai_codes'].apply(lambda codes_list: any(c in codes_list for c in selected_codes_for_filter))
                    filtered_view_df = st.session_state.processed_data[mask]
                    st.session_state.current_view_data = filtered_view_df
                    st.write(f"Showing {len(filtered_view_df)} items matching selected codes.")
                    display_cols = [st.session_state.get('col_for_coding', filtered_view_df.columns[0] if not filtered_view_df.empty else None), 'ai_codes']
                    display_cols = [col for col in display_cols if col and col in filtered_view_df.columns] # Ensure columns exist
                    if display_cols: st.dataframe(filtered_view_df[display_cols].head())

                    view_name_input = st.text_input("Save this filtered view as:", key="view_name_save_input")
                    if st.button("Save Current Filtered View", key="save_view_button_main"):
                        if view_name_input and not filtered_view_df.empty: data_management.save_view(filtered_view_df, view_name_input)
                        elif filtered_view_df.empty: ui_helpers.show_warning_message("Cannot save an empty view.")
                        else: ui_helpers.show_error_message("View name cannot be empty.")
                else: st.session_state.current_view_data = None; st.info("Select codes to create a filtered view, or load a saved view.")
            st.subheader("Load Saved View")
            saved_views_dict = data_management.load_saved_views()
            if saved_views_dict:
                view_names = [""] + list(saved_views_dict.keys())
                selected_view_name_load = st.selectbox("Choose a saved view to load:", options=view_names, key="load_saved_view_select")
                if st.button("Load Selected View", key="load_view_button_main") and selected_view_name_load:
                    try:
                        loaded_view_df = pd.read_csv(saved_views_dict[selected_view_name_load])
                        if 'ai_codes' in loaded_view_df.columns:
                             loaded_view_df['ai_codes'] = loaded_view_df['ai_codes'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else ([x] if pd.notna(x) else []))
                        st.session_state.current_view_data = loaded_view_df
                        ui_helpers.show_success_message(f"View '{selected_view_name_load}' loaded as Current View for Tab 4.")
                        st.dataframe(st.session_state.current_view_data.head())
                    except Exception as e: ui_helpers.show_error_message(f"Error loading view '{selected_view_name_load}': {e}")
            else: st.info("No saved views in the current project's 'views' folder.")
        else: st.info("Load/Process data with 'ai_codes' column to enable filtering/views.")

# ========================== TAB 4: Clustering, Theme Analysis & Visualization ==========================
with tab4:
    st.header("Clustering, Theme Analysis & Visualization")
    data_for_analysis, source_name = (st.session_state.current_view_data, "Current Filtered View (from Tab 3)") if st.session_state.current_view_data is not None and not st.session_state.current_view_data.empty else \
                                   (st.session_state.processed_data, "All Processed Data") if st.session_state.processed_data is not None and not st.session_state.processed_data.empty else \
                                   (None, "")
    if data_for_analysis is None: st.warning("No data for analysis. Load/process data in Tab 2, and optionally create/load a view in Tab 3.")
    else:
        st.info(f"Using data from: **{source_name}** ({len(data_for_analysis)} items).")
        if 'ai_codes' not in data_for_analysis.columns: st.warning(f"'ai_codes' column not found in {source_name}. Generate codes in Tab 3.")
        else:
            data_for_analysis['ai_codes'] = data_for_analysis['ai_codes'].apply(lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else (eval(x) if isinstance(x, str) and x.startswith('[') else [str(x)])))
            load_project_if_selected()
            st.subheader("Cluster Data Based on Codes (Conceptual)")
            st.markdown("Actual clustering requires libraries like `scikit-learn`. This section provides a **simulation**.")
            if st.button("Simulate Clustering (Assign Random Clusters)", key="simulate_cluster_button_main"):
                import random
                num_clusters_sim = st.slider("Number of demo clusters:", 2, 10, 3, key="num_sim_clusters_slider")
                data_for_analysis['cluster_label'] = [f"Cluster {random.randint(1, num_clusters_sim)}" for _ in range(len(data_for_analysis))]
                if data_for_analysis is st.session_state.current_view_data: st.session_state.current_view_data = data_for_analysis
                elif data_for_analysis is st.session_state.processed_data: st.session_state.processed_data = data_for_analysis; data_management.save_data_to_project(st.session_state.processed_data, "processed_data.csv")
                ui_helpers.show_success_message(f"Simulated clustering complete. 'cluster_label' added to {source_name}.")
                display_cols_clust = [st.session_state.get('col_for_coding', data_for_analysis.columns[0]), 'ai_codes', 'cluster_label']
                display_cols_clust = [col for col in display_cols_clust if col and col in data_for_analysis.columns]
                if display_cols_clust: st.dataframe(data_for_analysis[display_cols_clust].head())
            if 'cluster_label' in data_for_analysis.columns:
                st.subheader("AI-Driven Cluster Summaries")
                if not st.session_state.project_config.get('ai_provider') or not st.session_state.project_config.get(f"{st.session_state.project_config.get('ai_provider', '').lower()}_api", {}).get('api_key'): st.warning("AI Provider not configured (Tab 1).")
                else:
                    text_col_for_summary = st.session_state.get('col_for_coding')
                    if not text_col_for_summary or text_col_for_summary not in data_for_analysis.columns: st.error(f"Original text column ('{text_col_for_summary}') not found. Cannot summarize.")
                    else:
                        unique_clusters = sorted(data_for_analysis['cluster_label'].dropna().unique())
                        if not unique_clusters: st.info("No cluster labels to summarize.")
                        else:
                            selected_cluster_for_summary = st.selectbox("Select a cluster to summarize:", unique_clusters, key="cluster_summary_select")
                            if st.button(f"Generate AI Summary for {selected_cluster_for_summary}", key="summarize_cluster_button_main"):
                                cluster_texts_for_summary = data_for_analysis[data_for_analysis['cluster_label'] == selected_cluster_for_summary][text_col_for_summary].dropna().tolist()
                                if not cluster_texts_for_summary: ui_helpers.show_error_message(f"No texts in '{selected_cluster_for_summary}'.")
                                else:
                                    ai_provider_cfg, ai_model_cfg = st.session_state.project_config.get('ai_provider', "OpenAI"), None
                                    if ai_provider_cfg == "OpenAI": ai_model_cfg = "gpt-3.5-turbo"
                                    elif ai_provider_cfg == "Gemini": ai_model_cfg = "gemini-pro"
                                    if ai_model_cfg:
                                        summary = ai_services.generate_cluster_summary(cluster_texts_for_summary, ai_provider_cfg, model_name=ai_model_cfg)
                                        st.markdown(f"**AI-Generated Summary for {selected_cluster_for_summary}:**\n{summary}")
                                        if 'cluster_summaries' not in st.session_state: st.session_state.cluster_summaries = {}
                                        st.session_state.cluster_summaries[selected_cluster_for_summary] = summary
                                    else: st.error("AI model for summarization not determined.")
            else: st.info("Perform clustering to enable summaries.")
            st.header("Visualizations")
            st.subheader("Word Cloud")
            text_col_for_viz = st.session_state.get('col_for_coding')
            viz_source_options = (["From AI Codes"] if 'ai_codes' in data_for_analysis.columns else []) + \
                                 ([f"From Text Column ('{text_col_for_viz}')"] if text_col_for_viz and text_col_for_viz in data_for_analysis.columns else [])
            if not viz_source_options: st.info("No suitable data for word cloud.")
            else:
                viz_source_wc = st.radio("Generate word cloud from:", viz_source_options, horizontal=True, key="wc_source_radio")
                if st.button("Generate Word Cloud", key="wordcloud_button_main"):
                    text_for_wc_build = ""
                    if viz_source_wc == "From AI Codes": text_for_wc_build = " ".join([code for sublist in data_for_analysis['ai_codes'].dropna() for code in sublist if code])
                    elif viz_source_wc.startswith("From Text Column"): text_for_wc_build = " ".join(data_for_analysis[text_col_for_viz].dropna().astype(str).tolist())
                    if text_for_wc_build.strip():
                        try:
                            with ui_helpers.show_spinner("Generating word cloud..."):
                                wordcloud_img = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(text_for_wc_build)
                                fig_wc, ax_wc = plt.subplots(figsize=(10,5)); ax_wc.imshow(wordcloud_img, interpolation='bilinear'); ax_wc.axis('off'); st.pyplot(fig_wc)
                        except Exception as e: ui_helpers.show_error_message(f"Could not generate word cloud: {e}"); logger.error(f"Word cloud error: {e}")
                    else: ui_helpers.show_warning_message(f"No text/codes from '{viz_source_wc}' for word cloud.")
            st.subheader("Code Frequencies (Top 20)")
            if 'ai_codes' in data_for_analysis.columns:
                all_codes_flat_list_freq = [code for sublist in data_for_analysis['ai_codes'].dropna() for code in sublist if code]
                if all_codes_flat_list_freq:
                    code_counts_series = pd.Series(all_codes_flat_list_freq).value_counts().nlargest(20)
                    if not code_counts_series.empty: st.bar_chart(code_counts_series)
                    else: st.info("No codes to display frequencies for.")
                else: st.info("No codes found in data for frequencies.")
            else: st.info("Generate AI codes and ensure they are in the current analysis set for code frequencies.")

st.sidebar.info(
    """
    **Qualitative Thematic Analysis Tool**
    **Workflow:**
    1. **Tab 1:** Setup project, storage (Local/OneDrive), API keys.
    2. **Tab 2:** Fetch Reddit data, view, edit, redact.
    3. **Tab 3:** AI coding, filter data, manage views.
    4. **Tab 4:** (Simulate) Clustering, AI summaries, visualize.
    **Notes:** OneDrive uses Device Flow. Google Drive/Dropbox are conceptual.
    """
)
st.sidebar.markdown("---")
if st.session_state.current_project_name and st.session_state.project_path:
    st.sidebar.success(f"Active Project: {st.session_state.current_project_name}")
    project_sidebar_path_display = st.session_state.project_path
    if st.session_state.project_config.get('storage_type') != "Local":
        project_sidebar_path_display = f"{st.session_state.project_config.get('storage_type')} (Config: {st.session_state.project_path})"
    st.sidebar.caption(f"Location: {project_sidebar_path_display}")
else:
    st.sidebar.warning("No active project.")

if st.session_state.current_project_name and not st.session_state.project_config.get('project_id'):
    logger.info("Attempting to reload project state due to missing project_id...")
    if not st.session_state.project_path:
        project_id_guess = utils.generate_project_id(st.session_state.current_project_name)
        # This logic needs to be smarter if local path was user-defined.
        # For now, assumes if project_path is lost, it might be under default 'data' or user's home.
        # This part of session recovery is tricky without knowing the original user_chosen_local_base_path.
        # Best effort: try to load from a common default if absolutely no path info.
        st.session_state.project_path = os.path.join(st.session_state.project_config.get('user_local_base_path_for_projects', 'data'), project_id_guess)

    if st.session_state.project_path:
        load_project_if_selected()