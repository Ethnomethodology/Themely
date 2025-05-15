# pages/data_management.py
import streamlit as st
# Persist global theme and layout on this page
import pandas as pd
import os
from datetime import datetime
from modules import data_manager, ui_helpers, utils, reddit_api # Ensure all needed modules are imported

logger = utils.setup_logger("p01_data_management")
ui_helpers.page_sidebar_info([
    "Configure Reddit query",
    "Download data",
    "Combine datasets",
    "Create views"
])

# Helper function (could be moved to a shared utility if used by other pages too)
def load_and_combine_selected_downloads_for_view_p1(): # Renamed for page specificity
    combined_df_list = []
    # Ensure this session state key is initialized in app.py or here if not present
    if 'selected_download_files_info' not in st.session_state: st.session_state.selected_download_files_info = {}

    selected_file_infos = [ data for data in st.session_state.selected_download_files_info.values() if data.get("selected", False) ]
    if not selected_file_infos:
        st.session_state.combined_data_for_view_creation = pd.DataFrame()
        return
    for file_info in selected_file_infos:
        df_single = data_manager.load_data_from_specific_file(file_info["filepath"])
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
        st.session_state.combined_data_for_view_creation = concatenated_df # Store in global session state
    else: st.session_state.combined_data_for_view_creation = pd.DataFrame()


st.title("Data Management")
st.write("Manage your Reddit data: fetch posts, combine datasets, and create views for analysis.")

if not st.session_state.get('current_project_name'):
    st.warning("Please create or open a project on the Project Setup page.")
    st.stop()

# ... (Rest of the content from your previous `pages/1_💾_Data_Management.py` or Tab 2) ...
# Ensure all widget keys are unique for this page (e.g., suffixed with _p01 or _data_mgmt)
# Call load_and_combine_selected_downloads_for_view_p1() when selections change.
st.subheader("Download Reddit Data")
st.write("Fetch Reddit posts based on your query parameters.")
with st.expander("Fetch options", expanded=True):
    with st.form("reddit_query_form_p01_nav"): 
        fetch_subreddit_p01 = st.text_input("Subreddit", help="e.g., 'learnpython'")
        fetch_query_p01 = st.text_input("Search query (optional)")
        fetch_limit_p01 = st.number_input("Posts to fetch", 1, 1000, 25)
        fetch_sort_p01 = st.selectbox("Sort Order", ["relevance", "hot", "top", "new"], index=0 if fetch_query_p01 else 1)
        fetch_time_filter_p01 = st.selectbox("Time Filter (for search/top)", ["all", "year", "month", "week", "day", "hour"])
        submitted_fetch_p01 = st.form_submit_button("Fetch data")

    if submitted_fetch_p01:
        if fetch_subreddit_p01:
            if not st.session_state.project_config.get('reddit_api') or not st.session_state.project_config.get('reddit_api', {}).get('client_id'):
                ui_helpers.show_error_message("Reddit API keys not configured. Please set them on the Project Setup page.")
            else:
                with ui_helpers.show_spinner("Fetching data..."):
                    fetched_df_p01 = reddit_api.fetch_reddit_data(fetch_subreddit_p01, fetch_query_p01, fetch_limit_p01, sort=fetch_sort_p01, time_filter=fetch_time_filter_p01)
                if fetched_df_p01 is not None and not fetched_df_p01.empty:
                    current_ts_fetch_p01 = datetime.now()
                    fetch_params_meta_p01 = {"subreddit": fetch_subreddit_p01, "query": fetch_query_p01, "limit": fetch_limit_p01, "sort": fetch_sort_p01, "time_filter": fetch_time_filter_p01, "timestamp": current_ts_fetch_p01.isoformat()}
                    saved_filepath_fetch_p01 = data_manager.save_downloaded_reddit_data(fetched_df_p01, fetch_subreddit_p01, fetch_query_p01, fetch_params_meta_p01, current_ts_fetch_p01)
                    if saved_filepath_fetch_p01:
                        ui_helpers.show_success_message(f"Data saved to {os.path.basename(saved_filepath_fetch_p01)}")
                        if 'selected_download_files_info' in st.session_state: st.session_state.selected_download_files_info = {} 
                        st.rerun()
                    else: ui_helpers.show_error_message("Failed to save downloaded data.")
                elif fetched_df_p01 is not None: ui_helpers.show_warning_message("No data returned for your query.")
        else: ui_helpers.show_error_message("Subreddit name is required.")

st.divider()
st.subheader("Manage downloaded datasets")
downloaded_files_metadata_list_p01 = data_manager.list_downloaded_files_metadata()

if not downloaded_files_metadata_list_p01:
    st.info("No data downloaded yet.")
else:
    st.markdown("**Select datasets to combine**")
    display_meta_for_editor_p01 = []
    for meta_item_p01 in downloaded_files_metadata_list_p01:
        file_key_p01 = meta_item_p01["filename"]
        if file_key_p01 not in st.session_state.selected_download_files_info:
            st.session_state.selected_download_files_info[file_key_p01] = {"selected": False, "filepath": meta_item_p01["filepath"], "metadata": meta_item_p01}
        
        display_meta_for_editor_p01.append({
            "Select": st.session_state.selected_download_files_info[file_key_p01].get("selected", False),
            "File Name": meta_item_p01["filename"], "Subreddit": meta_item_p01.get("subreddit", "N/A"),
            "Query": meta_item_p01.get("query_used", "N/A"), "Downloaded": meta_item_p01.get("download_timestamp_str", "N/A")
        })
    
    meta_df_for_editor_p01 = pd.DataFrame(display_meta_for_editor_p01)
    edited_meta_df_downloads_p01 = st.data_editor(
        meta_df_for_editor_p01, disabled=[col for col in meta_df_for_editor_p01.columns if col != "Select"],
        key="downloaded_files_editor_key_p01_nav", hide_index=True, height=min(300, (len(meta_df_for_editor_p01) + 1) * 35 + 3)
    )

    if not meta_df_for_editor_p01.equals(edited_meta_df_downloads_p01):
        for idx_p01, editor_row_p01 in edited_meta_df_downloads_p01.iterrows():
            filename_editor_p01 = editor_row_p01["File Name"]
            if filename_editor_p01 in st.session_state.selected_download_files_info:
                st.session_state.selected_download_files_info[filename_editor_p01]["selected"] = editor_row_p01["Select"]
        load_and_combine_selected_downloads_for_view_p1() # Use page-specific or imported helper
        st.rerun()

    st.markdown("**Combined dataset**")
    combined_df_for_view_p01 = st.session_state.get('combined_data_for_view_creation', pd.DataFrame())

    if combined_df_for_view_p01 is not None and not combined_df_for_view_p01.empty:
        search_col_view_p01, redact_col_view_p01, redact_btn_view_p01 = st.columns([2,2,1])
        with search_col_view_p01:
            search_key_view_p01 = "search_combined_data_p01_nav"
            search_term_view_p01 = st.text_input("Search combined data:", value=st.session_state.search_term_combined_data_tab2, key=search_key_view_p01, on_change=lambda: setattr(st.session_state, 'search_term_combined_data_tab2', st.session_state[search_key_view_p01]))
            st.session_state.search_term_combined_data_tab2 = search_term_view_p01
        
        df_display_for_view_creation_p01 = combined_df_for_view_p01.copy()
        if search_term_view_p01:
            df_display_for_view_creation_p01 = df_display_for_view_creation_p01[df_display_for_view_creation_p01.astype(str).apply(lambda r: r.str.contains(search_term_view_p01, case=False, na=False).any(), axis=1)]

        with redact_col_view_p01:
            text_cols_redact_view_creation_p01 = [col for col in df_display_for_view_creation_p01.columns if df_display_for_view_creation_p01[col].dtype == 'object']
            col_to_redact_view_creation_p01 = st.selectbox("Redact column:", text_cols_redact_view_creation_p01, index=text_cols_redact_view_creation_p01.index('text') if 'text' in text_cols_redact_view_creation_p01 else 0, key="redact_col_select_p01_nav") if text_cols_redact_view_creation_p01 else None
        
        with redact_btn_view_p01:
            st.write("") 
            if col_to_redact_view_creation_p01 and st.button("Redact column", key="redact_button_p01_nav"):
                st.session_state.redact_confirm_data_management = True # Page specific confirm flag
        
        if st.session_state.get("redact_confirm_data_management", False) and col_to_redact_view_creation_p01:
            def perform_redaction_combined_view_p01():
                count = data_manager.redact_text_column_in_place(st.session_state.combined_data_for_view_creation, col_to_redact_view_creation_p01)
                msg = f"Redaction complete. {count} items processed." if count > 0 else "No items for redaction."
                ui_helpers.show_success_message(msg)
                st.session_state.redact_confirm_data_management = False
                st.rerun() 
            st.warning(f"Redact column '{col_to_redact_view_creation_p01}' in combined data?")
            confirm_cols_p01 = st.columns(2)
            if confirm_cols_p01[0].button("Confirm Redaction", key="confirm_redact_p01_nav"): perform_redaction_combined_view_p01()
            if confirm_cols_p01[1].button("Cancel Redaction", key="cancel_redact_p01_nav"): st.session_state.redact_confirm_data_management = False; st.rerun()

        if not df_display_for_view_creation_p01.empty: 
            st.dataframe(df_display_for_view_creation_p01, height=300)
            st.divider()
            st.markdown("**Create a view for analysis**")
            view_name_create_input_p01 = st.text_input("View name", key="view_name_create_p01_nav")
            if st.button("Create view", key="create_view_button_p01_nav"):
                if not view_name_create_input_p01: ui_helpers.show_error_message("View name cannot be empty.")
                else:
                    source_files_info_view_p01 = [data_info["metadata"]["filename"] for fn, data_info in st.session_state.selected_download_files_info.items() if data_info.get("selected", False)]
                    if data_manager.save_project_view(df_display_for_view_creation_p01, view_name_create_input_p01, source_filenames_info=source_files_info_view_p01):
                        ui_helpers.show_success_message(f"View '{view_name_create_input_p01}' created. Proceed to AI Coding.")
        elif search_term_view_p01: st.info("No data matches your search.")
        else: st.info("Combined dataset is empty.")
    elif any(data_info.get("selected", False) for data_info in st.session_state.selected_download_files_info.values()):
        st.info("Loading selected dataset...")
    else:
        st.info("Select datasets to combine above.")