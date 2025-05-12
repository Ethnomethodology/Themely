# pages/analysis_page.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from modules import data_management, ui_helpers, utils, ai_services
import json

logger = utils.setup_logger("p03_analysis")

# Dialog for inspecting code details
@st.dialog("Code Details", width="large")
def show_code_details_dialog():
    code = st.session_state.inspected_code
    df_all = st.session_state.analysis_table_df.copy()
    details_df = df_all[df_all['Codes'].apply(lambda s: code in robust_comma_string_to_list_analysis(s))]
    st.dataframe(details_df, use_container_width=True)
    if st.button("Close"):
        st.rerun()

# --- Initialize Session State Variables for this Page ---
if 'selected_analysis_views_info' not in st.session_state:
    st.session_state.selected_analysis_views_info = {}
if 'analysis_table_df' not in st.session_state:
    st.session_state.analysis_table_df = pd.DataFrame()
if 'created_code_groups' not in st.session_state: 
    st.session_state.created_code_groups = {} 

# Manual Edit Tab State
if 'manual_edit_selected_group_radio' not in st.session_state: 
    st.session_state.manual_edit_selected_group_radio = "Uncategorised Codes" 
if 'manual_edit_uncategorised_action_radio' not in st.session_state: 
    st.session_state.manual_edit_uncategorised_action_radio = "Create New Group"
if 'manual_edit_new_group_name_input' not in st.session_state:
    st.session_state.manual_edit_new_group_name_input = ""
if 'manual_edit_add_to_group_dropdown' not in st.session_state:
    st.session_state.manual_edit_add_to_group_dropdown = None
if 'manual_edit_code_search_input' not in st.session_state: 
    st.session_state.manual_edit_code_search_input = ""
# manual_edit_codes_checkboxes_state is still used for checkboxes NOT in a form (i.e., editing existing group)
if 'manual_edit_codes_checkboxes_state' not in st.session_state: 
    st.session_state.manual_edit_codes_checkboxes_state = {}


if 'selected_group_filter_analysis' not in st.session_state:
    st.session_state.selected_group_filter_analysis = "Show All"
if 'analysis_prompt_save_as_new_view' not in st.session_state:
    st.session_state.analysis_prompt_save_as_new_view = False
if 'analysis_new_view_name' not in st.session_state:
    st.session_state.analysis_new_view_name = ""
if 'ai_grouping_prompt_template_p03' not in st.session_state:
    st.session_state.ai_grouping_prompt_template_p03 = """You are an expert qualitative researcher.
Given the following list of unique thematic codes extracted from a dataset:
{unique_codes_list_json}

Your task is to group the codes into 3-10 coherent, non-overlapping themes.
For each theme (group), provide a concise "group_name" and list the specific "codes" from the input list that belong to this group.
A code MUST NOT belong to more than one group. Aim for distinct thematic clusters.

Return **only** a valid JSON array where each element is an object with two fields: "group_name" (string) and "codes" (an array of strings, where each string is one of the original codes provided).
Example Output Format:
[
  { "group_name": "Positive Experiences", "codes": ["good service", "helpful staff", "satisfied"] },
  { "group_name": "Technical Problems", "codes": ["login issue", "slow website", "error message"] }
]
Do not include any introductory text, explanations, or any characters outside the main JSON array structure.
"""
# TEXT_PREVIEW_COLUMNS removed as Column 3 is removed

# --- Helper Functions ---
def robust_comma_string_to_list_analysis(code_str):
    if pd.isna(code_str) or not str(code_str).strip(): return []
    if isinstance(code_str, list): 
        return [str(c).strip() for c in code_str if str(c).strip()]
    return [c.strip() for c in str(code_str).split(',') if c.strip()]

def robust_list_to_comma_string_analysis(code_list):
    if isinstance(code_list, list):
        return ", ".join(str(c).strip() for c in code_list if str(c).strip())
    if pd.isna(code_list) or not str(code_list).strip(): return ""
    return str(code_list).strip()

def load_and_combine_selected_analysis_views():
    combined_df_list = []
    st.session_state.analysis_source_view_paths = [] 
    selected_view_infos = [ data for data in st.session_state.selected_analysis_views_info.values() if data.get("selected", False)]
    if not selected_view_infos:
        st.session_state.analysis_table_df = pd.DataFrame()
        st.session_state.created_code_groups = {} 
        st.session_state.selected_group_filter_analysis = "Show All"
        st.session_state.manual_edit_selected_group_radio = "Uncategorised Codes" 
        return
    for view_info in selected_view_infos:
        csv_path = view_info["metadata"].get("csv_filepath")
        if csv_path and os.path.exists(csv_path):
            df_single_view = data_management.load_data_from_specific_file(csv_path)
            if df_single_view is not None:
                combined_df_list.append(df_single_view)
                st.session_state.analysis_source_view_paths.append(csv_path)
        else: logger.warning(f"CSV filepath missing or invalid for analysis view: {view_info['metadata'].get('view_name')}")
    if combined_df_list:
        concatenated_df = pd.concat(combined_df_list, ignore_index=True)
        id_cols_present = [col for col in ['unique_app_id', 'id'] if col in concatenated_df.columns]
        if id_cols_present:
            dedup_col = 'unique_app_id' if 'unique_app_id' in id_cols_present else id_cols_present[0]
            concatenated_df.drop_duplicates(subset=[dedup_col], keep='first', inplace=True)
        else: concatenated_df.drop_duplicates(keep='first', inplace=True)
        if 'Codes' not in concatenated_df.columns: concatenated_df['Codes'] = ""
        else: concatenated_df['Codes'] = concatenated_df['Codes'].apply(robust_list_to_comma_string_analysis)
        if 'groups' not in concatenated_df.columns: concatenated_df['groups'] = ""
        else: concatenated_df['groups'] = concatenated_df['groups'].apply(robust_list_to_comma_string_analysis)
        st.session_state.analysis_table_df = concatenated_df
        st.session_state.created_code_groups = {} 
        st.session_state.selected_group_filter_analysis = "Show All"
        st.session_state.manual_edit_selected_group_radio = "Uncategorised Codes"
    else:
        st.session_state.analysis_table_df = pd.DataFrame()
        st.session_state.created_code_groups = {}
        st.session_state.selected_group_filter_analysis = "Show All"
        st.session_state.manual_edit_selected_group_radio = "Uncategorised Codes"

def get_unique_codes_from_analysis_table():
    if st.session_state.analysis_table_df.empty or 'Codes' not in st.session_state.analysis_table_df.columns: return []
    all_codes_list = []
    for codes_str in st.session_state.analysis_table_df['Codes'].dropna():
        all_codes_list.extend(robust_comma_string_to_list_analysis(codes_str))
    return sorted(list(set(c for c in all_codes_list if c)))

def get_all_assigned_codes():
    assigned_codes = set()
    for group_name, codes_in_group in st.session_state.created_code_groups.items():
        assigned_codes.update(codes_in_group)
    return assigned_codes

def update_dataframe_groups_column():
    if st.session_state.analysis_table_df.empty: return
    df = st.session_state.analysis_table_df.copy()
    df['groups'] = '' 
    for group_name, codes_in_group_definition in st.session_state.created_code_groups.items():
        set_codes_in_group_def = set(codes_in_group_definition)
        for index, row in df.iterrows():
            row_codes = set(robust_comma_string_to_list_analysis(row['Codes']))
            if not row_codes.isdisjoint(set_codes_in_group_def):
                current_row_groups = robust_comma_string_to_list_analysis(df.loc[index, 'groups'])
                if group_name not in current_row_groups: 
                    current_row_groups.append(group_name)
                    df.loc[index, 'groups'] = robust_list_to_comma_string_analysis(current_row_groups)
    st.session_state.analysis_table_df = df

# get_texts_for_codes_preview function removed

def handle_checkbox_change(code_key, checkbox_key_in_session):
    # This function is now only relevant for checkboxes NOT inside a form
    # (i.e., when editing an existing group's codes)
    st.session_state.manual_edit_codes_checkboxes_state[code_key] = st.session_state[checkbox_key_in_session]

st.title("Analysis & Visualization")

if not st.session_state.get('current_project_name'):
    st.warning("👈 Please create or open a project first from the '🏠 Project Setup' page.")
    st.stop()

# --- 1. View Selection ---
st.subheader("Select Coded Project View(s) for Analysis")
available_views_meta_analysis = data_management.list_created_views_metadata()
if not available_views_meta_analysis:
    st.info("No project views created yet. Go to '💾 Data Management' to create views, and '🤖 AI Coding' to code them.")
else:
    display_views_for_analysis_editor = []
    for view_meta_item in available_views_meta_analysis:
        view_key = view_meta_item["view_name"] + "_analysis_select" 
        if view_key not in st.session_state.selected_analysis_views_info: 
            st.session_state.selected_analysis_views_info[view_key] = {"selected": False, "metadata": view_meta_item}
        display_views_for_analysis_editor.append({
            "Select": st.session_state.selected_analysis_views_info[view_key].get("selected", False), "View Name": view_meta_item["view_name"],
            "Created On": datetime.fromisoformat(view_meta_item.get("creation_timestamp", "")).strftime("%Y-%m-%d %H:%M") if view_meta_item.get("creation_timestamp") else "N/A",
            "Source Files Info": str(view_meta_item.get("source_files_info", "N/A")) 
        })
    views_df_for_analysis_editor = pd.DataFrame(display_views_for_analysis_editor)
    edited_views_df_analysis_selection = st.data_editor( views_df_for_analysis_editor, column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=[col for col in views_df_for_analysis_editor.columns if col != "Select"], key="analysis_views_selector_editor", hide_index=True, height=min(250, (len(views_df_for_analysis_editor) + 1) * 35 + 3))
    if not views_df_for_analysis_editor.equals(edited_views_df_analysis_selection):
        for idx, editor_row in edited_views_df_analysis_selection.iterrows():
            view_key_selection = editor_row["View Name"] + "_analysis_select"
            if view_key_selection in st.session_state.selected_analysis_views_info: st.session_state.selected_analysis_views_info[view_key_selection]["selected"] = editor_row["Select"]
        load_and_combine_selected_analysis_views(); st.rerun()

st.divider()

# --- 2. Analysis Table Display & Filter ---
st.subheader("Analysis Table")
analysis_df_master = st.session_state.get('analysis_table_df', pd.DataFrame()) 
displayed_analysis_df_view = analysis_df_master.copy()

if not analysis_df_master.empty:
    if st.session_state.created_code_groups:
        group_filter_options = ["Show All"] + sorted(list(st.session_state.created_code_groups.keys()))
        current_filter_index = group_filter_options.index(st.session_state.selected_group_filter_analysis) if st.session_state.selected_group_filter_analysis in group_filter_options else 0
        new_selected_filter = st.selectbox("Filter Displayed Table by Group:", options=group_filter_options, index=current_filter_index, key="group_filter_dropdown_analysis_main_key_above_table")
        if new_selected_filter != st.session_state.selected_group_filter_analysis: st.session_state.selected_group_filter_analysis = new_selected_filter; st.rerun() 
    else:
        st.caption("No code groups defined. Use tabs below to create groups.")
        if st.session_state.selected_group_filter_analysis != "Show All": st.session_state.selected_group_filter_analysis = "Show All"
    if st.session_state.selected_group_filter_analysis != "Show All" and st.session_state.selected_group_filter_analysis in st.session_state.created_code_groups:
        group_to_filter = st.session_state.selected_group_filter_analysis
        if 'groups' in displayed_analysis_df_view.columns:
            try: displayed_analysis_df_view = displayed_analysis_df_view[displayed_analysis_df_view['groups'].apply(lambda x: group_to_filter in robust_comma_string_to_list_analysis(x))]
            except Exception as e: logger.error(f"Error filtering by group: {e}"); st.error("Could not apply group filter.")
    # Calculate dynamic table width to allow horizontal scrolling
    table_width = min(displayed_analysis_df_view.shape[1] * 200, 2000)
    st.dataframe(displayed_analysis_df_view, height=350, width=table_width, key="analysis_table_display_key_main")
    col_spacer_save_analysis, col_save_analysis_button = st.columns([0.8, 0.2])
    with col_save_analysis_button:
        if st.button("Save Analysis Data", key="save_analysis_data_btn_main", use_container_width=True):
            if st.session_state.analysis_table_df.empty: ui_helpers.show_error_message("No analysis data to save.")
            else:
                source_paths = st.session_state.get('analysis_source_view_paths', []);
                if len(source_paths) == 1 and not st.session_state.analysis_prompt_save_as_new_view: st.session_state.confirm_overwrite_analysis_view_path = source_paths[0]
                else: st.session_state.analysis_prompt_save_as_new_view = True
                st.rerun()
    if 'confirm_overwrite_analysis_view_path' in st.session_state and st.session_state.confirm_overwrite_analysis_view_path:
        path_to_overwrite = st.session_state.confirm_overwrite_analysis_view_path; st.warning(f"Overwrite '{os.path.basename(path_to_overwrite)}'?")
        col_confirm, col_cancel = st.columns(2)
        if col_confirm.button("Yes, Overwrite", key="confirm_overwrite_analysis_save_main"):
            df_to_save = st.session_state.analysis_table_df.copy(); data_management.save_coded_data_to_view(df_to_save, path_to_overwrite)
            ui_helpers.show_success_message(f"Saved to '{os.path.basename(path_to_overwrite)}'."); del st.session_state['confirm_overwrite_analysis_view_path']; st.rerun()
        if col_cancel.button("No, Cancel", key="cancel_overwrite_analysis_save_main"): del st.session_state['confirm_overwrite_analysis_view_path']; st.session_state.analysis_prompt_save_as_new_view = True; st.rerun()
    if st.session_state.get("analysis_prompt_save_as_new_view", False):
        st.info("Save current Analysis Table as new Project View.")
        col_new_name_analysis, col_new_save_analysis = st.columns([0.7, 0.3]);
        with col_new_name_analysis: st.session_state.analysis_new_view_name = st.text_input("Enter Name for New Analysis View:", value=st.session_state.analysis_new_view_name, key="analysis_new_view_name_input_form_key_main")
        with col_new_save_analysis:
            st.write("") 
            if st.button("Confirm Save New", key="analysis_save_new_key_main", use_container_width=True):
                new_name = st.session_state.analysis_new_view_name.strip()
                if not new_name: ui_helpers.show_error_message("Name empty.")
                else:
                    df_to_save = st.session_state.analysis_table_df.copy(); sources = [os.path.basename(p) for p in st.session_state.get('analysis_source_view_paths', [])] or ["analysis session"]
                    if data_management.save_project_view(df_to_save, new_name, source_filenames_info=sources):
                        ui_helpers.show_success_message(f"Saved as '{new_name}'."); st.session_state.analysis_prompt_save_as_new_view = False; st.session_state.analysis_new_view_name = ""; st.rerun()
                    else: ui_helpers.show_error_message("Save failed.")
    st.divider()
    st.subheader("Code Grouping Actions")
    st.caption("Define non-overlapping groups. A code can only belong to one group.")
    tab_ai_group, tab_manual_edit_group = st.tabs(["Suggest Groups with AI", "Manually edit groups"])
    with tab_ai_group:
        st.markdown("Let AI suggest thematic groups based on **unassigned** codes in your current Analysis Table.")
        ai_provider_analysis = st.session_state.project_config.get('ai_provider', "OpenAI"); ai_api_key_present = st.session_state.project_config.get(f"{ai_provider_analysis.lower()}_api", {}).get('api_key')
        if not ai_api_key_present: st.warning(f"{ai_provider_analysis} API key not configured in Project Setup.")
        else:
            st.info(f"Using AI Provider: **{ai_provider_analysis}**"); ai_model_analysis = None
            if ai_provider_analysis == "OpenAI": ai_model_analysis = st.selectbox("OpenAI Model:", ["gpt-4o", "gpt-4-turbo","gpt-3.5-turbo"], key="ai_model_group_openai_bottom")
            elif ai_provider_analysis == "Gemini": ai_model_analysis = st.selectbox("Gemini Model:", ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.0-pro"], key="ai_model_group_gemini_bottom")
            all_unique_codes = get_unique_codes_from_analysis_table(); codes_already_assigned_ai = get_all_assigned_codes(); unassigned_codes_for_ai = sorted(list(set(all_unique_codes) - codes_already_assigned_ai))
            st.session_state.ai_grouping_prompt_template_p03 = st.text_area("AI Grouping Prompt:", value=st.session_state.ai_grouping_prompt_template_p03, key="ai_group_prompt_input_bottom", height=300)
            if not unassigned_codes_for_ai: st.info("All unique codes are already assigned to groups.")
            elif st.button("Generate Groups with AI", key="ai_generate_groups_btn_bottom", disabled=(not ai_model_analysis)):
                if '{unique_codes_list_json}' not in st.session_state.ai_grouping_prompt_template_p03: ui_helpers.show_error_message("Prompt needs '{unique_codes_list_json}'.")
                else:
                    unassigned_codes_json_str = json.dumps(unassigned_codes_for_ai, indent=2)
                    with st.spinner(f"Sending {len(unassigned_codes_for_ai)} codes to AI..."):
                        ai_response_groups = ai_services.generate_code_groups_with_ai(unassigned_codes_json_str, st.session_state.ai_grouping_prompt_template_p03, ai_provider_analysis, ai_model_analysis)
                    if isinstance(ai_response_groups, list) and all(isinstance(g, dict) and "group_name" in g and "codes" in g for g in ai_response_groups):
                        new_groups_ai_count = 0; temp_assigned_ai_run = set()
                        for group_data in ai_response_groups:
                            ai_g_name = str(group_data.get("group_name", "AI_Group")).strip(); ai_g_codes = [str(c).strip() for c in group_data.get("codes", []) if str(c).strip()]
                            if not ai_g_name or not ai_g_codes: logger.warning(f"AI empty group/codes: {group_data}"); continue
                            valid_codes_new_ai_g = [c for c in ai_g_codes if c in unassigned_codes_for_ai and c not in codes_already_assigned_ai and c not in temp_assigned_ai_run]
                            if not valid_codes_new_ai_g: logger.info(f"AI group '{ai_g_name}' no valid unassigned codes. Skipping."); continue
                            final_ai_g_name_unique = ai_g_name; c_u = 1
                            while final_ai_g_name_unique in st.session_state.created_code_groups: final_ai_g_name_unique = f"{ai_g_name}_{c_u}"; c_u += 1
                            st.session_state.created_code_groups[final_ai_g_name_unique] = valid_codes_new_ai_g
                            codes_already_assigned_ai.update(valid_codes_new_ai_g); temp_assigned_ai_run.update(valid_codes_new_ai_g); new_groups_ai_count += 1
                        if new_groups_ai_count > 0: update_dataframe_groups_column(); ui_helpers.show_success_message(f"AI suggested {new_groups_ai_count} new groups.")
                        else: ui_helpers.show_info_message("AI complete. No new valid non-overlapping groups formed.")
                        st.rerun()
                    else: err_d = str(ai_response_groups) if ai_response_groups else "No response."; ui_helpers.show_error_message(f"AI failed. Details: {err_d[:200]}..."); logger.error(f"AI group failed: {ai_response_groups}")
    
    with tab_manual_edit_group:
        st.header("Manage Group Assignments")
        # Layout changes to 2 columns
        col1_manual, col2_manual = st.columns([0.35, 0.65]) 
        all_db_unique_codes = get_unique_codes_from_analysis_table()
        assigned_codes_globally = get_all_assigned_codes()
        uncategorised_codes_list = sorted(list(set(all_db_unique_codes) - assigned_codes_globally))
        
        with col1_manual:
            st.subheader("Select")
            group_names_for_radio = sorted(list(st.session_state.created_code_groups.keys()))
            uncategorised_label = f"Uncategorised Codes ({len(uncategorised_codes_list)})"
            display_radio_options = [uncategorised_label] + group_names_for_radio
            current_radio_selection_display = st.session_state.get('manual_edit_selected_group_radio', uncategorised_label) 
            if current_radio_selection_display == "Uncategorised Codes" and current_radio_selection_display != uncategorised_label : current_radio_selection_display = uncategorised_label
            
            try: current_radio_idx = display_radio_options.index(current_radio_selection_display)
            except ValueError: current_radio_idx = 0; st.session_state.manual_edit_selected_group_radio = "Uncategorised Codes"

            selected_radio_display_val = st.radio("Edit codes for:", options=display_radio_options, index=current_radio_idx, key="manual_edit_group_selector_radio_key_col1",
                                                 on_change=lambda: st.session_state.update({'manual_edit_codes_checkboxes_state': {}, 'manual_edit_code_search_input': ''})) 
            
            if selected_radio_display_val == uncategorised_label: st.session_state.manual_edit_selected_group_radio = "Uncategorised Codes"
            else: st.session_state.manual_edit_selected_group_radio = selected_radio_display_val
        
        with col2_manual:
            st.subheader("Codes")
            current_selection_context = st.session_state.manual_edit_selected_group_radio
            
            if current_selection_context == "Uncategorised Codes":
                st.session_state.manual_edit_uncategorised_action_radio = st.radio(
                    "Action for selected uncategorised codes:", 
                    ["Create New Group", "Add to Existing Group"], 
                    key="uncat_action_radio_key_col2", horizontal=True, 
                    index=0 if st.session_state.manual_edit_uncategorised_action_radio == "Create New Group" else 1,
                    on_change=lambda: st.session_state.update({'manual_edit_codes_checkboxes_state': {}, 'manual_edit_code_search_input': '', 'manual_edit_new_group_name_input': ''}) 
                )
            
            codes_to_display_checkboxes = [] 
            
            if current_selection_context == "Uncategorised Codes":
                if st.session_state.manual_edit_uncategorised_action_radio == "Create New Group":
                    # Custom group creation UI to allow Inspect buttons
                    st.session_state.manual_edit_new_group_name_input = st.text_input(
                        "New Group Name:",
                        value=st.session_state.manual_edit_new_group_name_input,
                        key="col2_new_group_name"
                    )
                    st.session_state.manual_edit_code_search_input = st.text_input(
                        "Search uncategorised codes:",
                        value=st.session_state.manual_edit_code_search_input,
                        key="col2_search_uncat_for_new"
                    )
                    codes_to_display_checkboxes = [c for c in uncategorised_codes_list if st.session_state.manual_edit_code_search_input.lower() in c.lower()] if st.session_state.manual_edit_code_search_input else uncategorised_codes_list
                    st.caption("Select uncategorised codes for this new group:")
                    # Collect selected codes and provide Inspect buttons
                    selected_codes = []
                    with st.container(height=300):
                        for code in codes_to_display_checkboxes:
                            col_cb, col_btn = st.columns([0.75, 0.25])
                            checked = col_cb.checkbox(
                                code,
                                value=st.session_state.manual_edit_codes_checkboxes_state.get(code, False),
                                key=f"cb_new_{utils.generate_project_id(code)}"
                            )
                            if checked:
                                selected_codes.append(code)
                            if col_btn.button("Inspect", key=f"inspect_{utils.generate_project_id(code)}"):
                                st.session_state.inspected_code = code
                                show_code_details_dialog()
                        st.markdown('</div>', unsafe_allow_html=True)
                    # Finalize group creation
                    if st.button("Create Group"):
                        new_g_name = st.session_state.manual_edit_new_group_name_input.strip()
                        if not new_g_name:
                            ui_helpers.show_error_message("Name empty.")
                        elif not selected_codes:
                            ui_helpers.show_error_message("Select codes.")
                        elif new_g_name in st.session_state.created_code_groups:
                            ui_helpers.show_error_message(f"Group '{new_g_name}' exists.")
                        else:
                            st.session_state.created_code_groups[new_g_name] = selected_codes.copy()
                            update_dataframe_groups_column()
                            ui_helpers.show_success_message(f"Group '{new_g_name}' created.")
                            st.session_state.manual_edit_new_group_name_input = ""
                            st.session_state.manual_edit_codes_checkboxes_state = {}
                            st.rerun()
                
                elif st.session_state.manual_edit_uncategorised_action_radio == "Add to Existing Group":
                    existing_g_names = sorted(list(st.session_state.created_code_groups.keys()))
                    if not existing_g_names: st.warning("No existing groups. Create one first.")
                    else:
                        with st.form("col2_add_to_existing_form"):
                            # Order: Target Group, then Search, then Checkboxes
                            st.session_state.manual_edit_add_to_group_dropdown = st.selectbox("Target Group:", options=existing_g_names, key="col2_add_to_group_dd", index=existing_g_names.index(st.session_state.manual_edit_add_to_group_dropdown) if st.session_state.manual_edit_add_to_group_dropdown in existing_g_names else 0)
                            st.session_state.manual_edit_code_search_input = st.text_input("Search uncategorised codes:", value=st.session_state.manual_edit_code_search_input, key="col2_search_uncat_for_existing")
                            
                            codes_to_display_checkboxes = [c for c in uncategorised_codes_list if st.session_state.manual_edit_code_search_input.lower() in c.lower()] if st.session_state.manual_edit_code_search_input else uncategorised_codes_list
                            st.caption("Select uncategorised codes to add:")
                            current_form_checkbox_selections = {}
                            with st.container(height=150):
                                for code in codes_to_display_checkboxes: 
                                    current_form_checkbox_selections[code] = st.checkbox(code, value=st.session_state.manual_edit_codes_checkboxes_state.get(code,False), key=f"form_cb_uncat_add_{utils.generate_project_id(code)}_col2")
                            if st.form_submit_button("Add to Group"):
                                target_g = st.session_state.manual_edit_add_to_group_dropdown; selected_c_add = [c for c,chk in current_form_checkbox_selections.items() if chk]
                                if not target_g: ui_helpers.show_error_message("Select target group.")
                                elif not selected_c_add: ui_helpers.show_error_message("Select codes.")
                                else: 
                                    st.session_state.created_code_groups[target_g].extend(c for c in selected_c_add if c not in st.session_state.created_code_groups[target_g]); 
                                    st.session_state.created_code_groups[target_g] = sorted(list(set(st.session_state.created_code_groups[target_g]))); 
                                    update_dataframe_groups_column(); 
                                    ui_helpers.show_success_message(f"Codes added to '{target_g}'."); 
                                    st.session_state.manual_edit_codes_checkboxes_state = {}; 
                                    st.session_state.manual_edit_selected_group_radio = target_g; 
                                    st.rerun()
            
            elif current_selection_context in st.session_state.created_code_groups: 
                # Order: Search, then Checkboxes (no group name input here)
                st.session_state.manual_edit_code_search_input = st.text_input(f"Search codes within '{current_selection_context}':", value=st.session_state.manual_edit_code_search_input, key="col2_search_in_selected_group")
                codes_in_selected_group = sorted(st.session_state.created_code_groups[current_selection_context])
                codes_to_display_checkboxes = [c for c in codes_in_selected_group if st.session_state.manual_edit_code_search_input.lower() in c.lower()] if st.session_state.manual_edit_code_search_input else codes_in_selected_group
                
                st.caption(f"Codes in '{current_selection_context}'. Uncheck to remove from this group upon update.")
                with st.container(height=200):
                    for code in codes_to_display_checkboxes: 
                        cb_key = f"cb_existing_g_{utils.generate_project_id(code)}_{utils.generate_project_id(current_selection_context)}_col2"
                        st.checkbox(code, value=st.session_state.manual_edit_codes_checkboxes_state.get(code, True), 
                                    key=cb_key, on_change=handle_checkbox_change, args=(code, cb_key)) 
                if st.button(f"Update Group '{current_selection_context}' (Remove Unchecked)", key="col2_update_group_btn"):
                    codes_to_keep = [c for c in codes_in_selected_group if st.session_state.manual_edit_codes_checkboxes_state.get(c, False)] 
                    original_codes_this_group = set(st.session_state.created_code_groups[current_selection_context]); removed_count = len(original_codes_this_group - set(codes_to_keep))
                    if not codes_to_keep: del st.session_state.created_code_groups[current_selection_context]; ui_helpers.show_success_message(f"Group '{current_selection_context}' removed."); st.session_state.manual_edit_selected_group_radio = "Uncategorised Codes"
                    else: st.session_state.created_code_groups[current_selection_context] = codes_to_keep; ui_helpers.show_success_message(f"Group '{current_selection_context}' updated. {removed_count} code(s) uncategorised.")
                    update_dataframe_groups_column(); st.session_state.manual_edit_codes_checkboxes_state = {}; st.rerun() 
            else: st.empty() # Should not be reached if radio state is managed
else: 
    st.info("Select one or more views to load data for analysis and grouping.")
