# pages/themes.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from modules import data_manager, ui_helpers, utils, ai_services
import json

logger = utils.setup_logger("p03_analysis")
ui_helpers.page_sidebar_info([
    "Select Coded View(s) to list their record",
    "Cluster Groups of your selected view(s) using AI",
    "Review AI-generated Groups",
    "Save groups to the dataset once verified"
])


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
CURRENT GROUPS (Draft)
The JSON array below is your existing themes (may be empty if none):
{existing_groups_json}

Given the following array of code metadata extracted from the codebook:
{unique_codes_list_json}

Your task is to group the codes into 3-10 coherent, non-overlapping themes.
For each theme (group), provide a concise "group_name" and list the specific "codes" from the input list that belong to this group.
• Do NOT modify, merge, abbreviate, or invent any code names. Each entry in "codes" must exactly match one of the items in the provided list.
A code MUST NOT belong to more than one group. Aim for distinct thematic clusters.

RULES
• Only use codes that appear in the input list; do not include any others.
• Preserve exact spelling and capitalization of code names.
• Do NOT invent new codes or synonyms.
Return **only** a valid JSON array where each element is an object with three fields: 
  • "group_name" (string)
  • "codes" (an array of strings, where each string is one of the original codes provided)
  • "code_ids" (an array of strings, where each string is the corresponding code_id for that code)
Example Output Format:
[
  { "group_name": "Positive Experiences", 
    "codes": ["good service", "helpful staff", "satisfied"],
    "code_ids": ["uvea34462fji", "ihjdse3663g3", "fejyrse5426gsa"] 
  },
  { "group_name": "Technical Problems", 
    "codes": ["login issue", "slow website", "error message"],
    "code_ids": ["466huyuw35c", "ojht45wwt44", "gar3335hjyk32"] 
  }
]
Do not include any introductory text, explanations, or any characters outside the main JSON array structure.
"""

# Track last selected views for analysis to preserve manual groups across reruns
if 'last_analysis_selected_views' not in st.session_state:
    st.session_state['last_analysis_selected_views'] = []

# Track groups that have been saved to file
if 'saved_code_groups' not in st.session_state:
    st.session_state.saved_code_groups = {}

# Ensure we only load the groups overview CSV once per session to avoid overwriting in-memory group edits
if 'groups_overview_loaded' not in st.session_state:
    st.session_state['groups_overview_loaded'] = False

# Load persisted groups overview and initialize created_code_groups (but only once per session)
groups_overview_path = None
if st.session_state.get("project_path"):
    groups_dir = os.path.join(st.session_state.project_path, "groups")
    groups_overview_path = os.path.join(groups_dir, "groups_overview.csv")
if not st.session_state['groups_overview_loaded']:
    if groups_overview_path and os.path.exists(groups_overview_path):
        df_overview = pd.read_csv(groups_overview_path)
        st.session_state.groups_overview_df = df_overview
        # Populate created_code_groups and saved_code_groups from overview
        st.session_state.created_code_groups = {}
        st.session_state.saved_code_groups = {}
        for _, row in df_overview.iterrows():
            grp = row["Group Name"]
            code = row["Code Name"]
            st.session_state.created_code_groups.setdefault(grp, []).append(code)
            st.session_state.saved_code_groups.setdefault(grp, []).append(code)
    else:
        st.session_state.groups_overview_df = pd.DataFrame()
    st.session_state['groups_overview_loaded'] = True

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
    selected_infos = [info for info in st.session_state.selected_analysis_views_info.values() if info.get("selected", False)]
    current_views = [v["metadata"]["view_name"] for v in selected_infos]
    if st.session_state.get("last_analysis_selected_views") != current_views:
        st.session_state["last_analysis_selected_views"] = current_views
        if not selected_infos:
            st.session_state.analysis_table_df = pd.DataFrame()
            # st.session_state.created_code_groups = {}  # Do not clear created_code_groups when no views are selected
            st.session_state.selected_group_filter_analysis = "Show All"
            st.session_state.manual_edit_selected_group_radio = "Uncategorised Codes"
            return
        dfs = []
        st.session_state.analysis_source_view_paths = []
        for info in selected_infos:
            path = info["metadata"].get("csv_filepath")
            if path and os.path.exists(path):
                df = data_manager.load_data_from_specific_file(path)
                if df is not None:
                    dfs.append(df)
                    st.session_state.analysis_source_view_paths.append(path)
            else:
                logger.warning(f"Invalid CSV path for view: {info['metadata'].get('view_name')}")
        if dfs:
            df_all = pd.concat(dfs, ignore_index=True)
            for col in ['unique_app_id','id']:
                if col in df_all.columns:
                    df_all.drop_duplicates(subset=[col], keep='first', inplace=True)
                    break
            if 'Codes' in df_all.columns:
                df_all['Codes'] = df_all['Codes'].apply(robust_list_to_comma_string_analysis)
            else:
                df_all['Codes'] = ""
            if 'groups' in df_all.columns:
                df_all['groups'] = df_all['groups'].apply(robust_list_to_comma_string_analysis)
            else:
                df_all['groups'] = ""
            st.session_state.analysis_table_df = df_all
            # Apply group definitions to the loaded data immediately
            update_dataframe_groups_column()
        else:
            st.session_state.analysis_table_df = pd.DataFrame()
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

# --- AI Grouping JSON Inspection Dialog ---
@st.dialog("AI Raw Grouping JSON")
def ai_group_json_dialog():
    st.title("AI Raw Grouping JSON")
    json_str = st.session_state.get("last_ai_grouping_json", "")
    if json_str:
        st.code(json_str, language="json")
    else:
        st.info("No AI JSON available.")
    if st.button("Close", key="close_ai_group_json_dialog"):
        if "last_ai_grouping_json" in st.session_state:
            del st.session_state["last_ai_grouping_json"]
        st.rerun()

def handle_checkbox_change(code_key, checkbox_key_in_session):
    st.session_state.manual_edit_codes_checkboxes_state[code_key] = st.session_state[checkbox_key_in_session]

st.title("Themes")

if not st.session_state.get('current_project_name'):
    st.warning("👈 Please create or open a project first from the '🏠 Project Setup' page.")
    st.stop()

if 'edited_codebook_df' not in st.session_state or st.session_state.get('edited_codebook_df').empty:
    loaded_cb_df = data_manager.load_codebook(st.session_state.project_path)
    if "Select" not in loaded_cb_df.columns: # Ensure 'Select' column if loading fresh
        loaded_cb_df.insert(0, "Select", False)
    st.session_state.current_codebook_df = loaded_cb_df.copy()
    st.session_state.edited_codebook_df = loaded_cb_df.copy()

# --- 1. View Selection ---
st.subheader("Select Coded View(s) for Analysis")
available_views_meta_analysis = data_manager.list_created_views_metadata()
if not available_views_meta_analysis:
    st.info("No project views created yet. Go to '💾 Data Management' to create views, and '🤖 AI Coding' to code them.")
else:
    for vm in available_views_meta_analysis:
        view_key = vm["view_name"] + "_analysis_select"
        if view_key not in st.session_state.selected_analysis_views_info:
            st.session_state.selected_analysis_views_info[view_key] = {"selected": False, "metadata": vm}
    views_selection_df = pd.DataFrame([
        {
            "View Name": vm["view_name"],
            "Created On": datetime.fromisoformat(vm.get("creation_timestamp", "")).strftime("%Y-%m-%d %H:%M") if vm.get("creation_timestamp") else "N/A",
            "Source Files Info": ", ".join(vm.get("source_files_info") if isinstance(vm.get("source_files_info"), list) else [str(vm.get("source_files_info", ""))])
        }
        for vm in available_views_meta_analysis
    ])
    view_selection_event = st.dataframe(
        views_selection_df,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="multi-row"
    )
    selected_idxs = view_selection_event.selection.rows if hasattr(view_selection_event, 'selection') else []
    for idx, vm in enumerate(available_views_meta_analysis):
        key = vm["view_name"] + "_analysis_select"
        st.session_state.selected_analysis_views_info[key]["selected"] = idx in selected_idxs
    load_and_combine_selected_analysis_views()

st.divider()

# --- 2. Analysis Table Display & Filter ---
st.subheader("Coded Views")
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

    df_event = st.dataframe(
        displayed_analysis_df_view,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="multi-row"
    )
    selected_rows = df_event.selection.rows if hasattr(df_event, 'selection') else []
    st.session_state.analysis_selected_rows = selected_rows

    # --- END Save Analysis Data button block removed ---

    st.divider()

    # --- Groups Overview ---
    st.subheader("Groups Overview")
    group_rows = []
    for group_name, codes in st.session_state.created_code_groups.items():
        for code in codes:
            group_rows.append({"Group Name": group_name, "Code Name": code})
    groups_flat_df = pd.DataFrame(group_rows)

    codebook_source_df = st.session_state.get('edited_codebook_df', pd.DataFrame(columns=["Select"] + data_manager.CODEBOOK_COLUMNS))
    codebook_meta = codebook_source_df.drop(columns=["Select"], errors="ignore").copy() # Use .copy()

    if not groups_flat_df.empty:
        if "Code Name" in codebook_meta.columns and not codebook_meta.empty:
             groups_flat_df = groups_flat_df.merge(codebook_meta, on="Code Name", how="left")
        else:
            logger.info("Codebook metadata ('edited_codebook_df') is empty or missing 'Code Name' column. Metadata columns in Groups Overview will be blank if not already present in groups_flat_df.")
            for col_name in ["Description", "Rationale", "Example_ids", "code_id"]:
                if col_name not in groups_flat_df.columns:
                    groups_flat_df[col_name] = pd.NA

    if not groups_flat_df.empty:
        df_display = groups_flat_df.copy()
        
        display_columns_ordered = ["Group Name", "Code Name"]
        codebook_metadata_to_display = ["Description", "Rationale", "Example_ids", "code_id"]
        
        for meta_col in codebook_metadata_to_display:
            if meta_col in df_display.columns:
                display_columns_ordered.append(meta_col)
            elif meta_col in data_manager.CODEBOOK_COLUMNS and meta_col not in df_display.columns:
                df_display[meta_col] = pd.NA 
                display_columns_ordered.append(meta_col)

        remaining_cols = [col for col in df_display.columns if col not in display_columns_ordered]
        display_columns_ordered.extend(remaining_cols)
        
        final_display_columns = []
        seen_cols_display = set()
        for col in display_columns_ordered:
            if col in df_display.columns and col not in seen_cols_display:
                final_display_columns.append(col)
                seen_cols_display.add(col)
        
        df_display = df_display[final_display_columns].reset_index(drop=True)
        df_display.insert(0, "Serial No.", range(1, len(df_display) + 1))

        # Fill NaN with "None" for display
        df_display_filled = df_display.fillna("None")

        # Determine which (group, code) pairs are new or removed
        saved_pairs = {
            (grp, code)
            for grp, codes in st.session_state.saved_code_groups.items()
            for code in codes
        }
        created_pairs = {
            (grp, code)
            for grp, codes in st.session_state.created_code_groups.items()
            for code in codes
        }
        new_pairs = created_pairs - saved_pairs
        removed_pairs = saved_pairs - created_pairs
        styled_df = df_display_filled.style.apply(
            lambda row: [
                "background-color: #e6f2ff"
                if (row["Group Name"], row["Code Name"]) in new_pairs or
                   (row["Group Name"], row["Code Name"]) in removed_pairs
                else ""
                for _ in row
            ],
            axis=1
        )
        event_groups = st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            height=min(400, (len(df_display_filled) + 1) * 35 + 3)
        )
        st.session_state.analysis_group_selection = event_groups.selection.rows if hasattr(event_groups, "selection") else []

        # --- Save Groups Button ---
        col_spacer_groups, col_save_groups = st.columns([0.8, 0.2])
        with col_save_groups:
            if st.button("Save Groups", key="save_groups_btn", use_container_width=True):
                # Apply grouped labels to the table
                update_dataframe_groups_column()
                # Persist to all selected view files
                paths = st.session_state.get('analysis_source_view_paths', [])
                for path in paths:
                    df_to_save = st.session_state.analysis_table_df.copy()
                    if 'Source View' in df_to_save.columns:
                        df_to_save = df_to_save.drop(columns=['Source View'], errors='ignore')
                    success = data_manager.save_coded_data_to_view(df_to_save, path)
                    if success:
                        ui_helpers.show_success_message(f"Groups saved to '{os.path.basename(path)}'.")
                    else:
                        ui_helpers.show_error_message(f"Failed to save groups to '{os.path.basename(path)}'.")
                # Persist master groups overview
                groups_dir = os.path.join(st.session_state.project_path, "groups")
                os.makedirs(groups_dir, exist_ok=True)
                # Build overview DataFrame
                overview_rows = []
                for grp, codes in st.session_state.created_code_groups.items():
                    for c in codes:
                        overview_rows.append({"Group Name": grp, "Code Name": c})
                df_overview = pd.DataFrame(overview_rows)
                # Merge in codebook metadata for full overview
                codebook_meta = st.session_state.edited_codebook_df.drop(columns=["Select"], errors="ignore")
                df_overview_full = df_overview.merge(codebook_meta, on="Code Name", how="left")
                overview_path = os.path.join(groups_dir, "groups_overview.csv")
                df_overview_full.to_csv(overview_path, index=False)
                # Mark groups as saved
                st.session_state.saved_code_groups = st.session_state.created_code_groups.copy()
                st.rerun()
        st.divider()
    else:
        st.info("No groups defined yet, or no codes assigned to groups.")



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
            elif ai_provider_analysis == "Gemini": ai_model_analysis = st.selectbox("Gemini Model:", ["gemini-2.0-flash", "gemini-1.5-flash"], key="ai_model_group_gemini_bottom")
            # Pull all codes from the master codebook
            cb_df = st.session_state.edited_codebook_df
            all_codes = cb_df["Code Name"].dropna().unique().tolist()
            unassigned_codes_for_ai = sorted(all_codes)
            st.session_state.ai_grouping_prompt_template_p03 = st.text_area("AI Grouping Prompt:", value=st.session_state.ai_grouping_prompt_template_p03, key="ai_group_prompt_input_bottom", height=300)
            if not unassigned_codes_for_ai: st.info("All unique codes are already assigned to groups.")
            elif st.button("Generate Groups with AI", key="ai_generate_groups_btn_bottom", disabled=(not ai_model_analysis)):
                if '{unique_codes_list_json}' not in st.session_state.ai_grouping_prompt_template_p03: ui_helpers.show_error_message("Prompt needs '{unique_codes_list_json}'.")
                else:
                    # Build JSON for existing groups
                    existing = []
                    cb_meta = st.session_state.edited_codebook_df
                    for grp, codes in st.session_state.created_code_groups.items():
                        ids = []
                        for code in codes:
                            match = cb_meta[cb_meta["Code Name"] == code]
                            if not match.empty:
                                ids.append(str(match.iloc[0]["code_id"]))
                        existing.append({"group_name": grp, "codes": codes, "code_ids": ids})
                    existing_groups_json_str = json.dumps(existing, indent=2)
                    # Build metadata array for unassigned codes
                    code_metadata_list = []
                    cb_meta = st.session_state.edited_codebook_df.drop(columns=["Select"], errors="ignore")
                    for code in unassigned_codes_for_ai:
                        meta = cb_meta[cb_meta["Code Name"] == code]
                        if not meta.empty:
                            row = meta.iloc[0]
                            # Extract example IDs and map them to text from the analysis table
                            raw_ids = [s.strip() for s in str(row.get("Example_ids", "")).split(",") if s.strip()]
                            df_analysis = st.session_state.analysis_table_df
                            example_texts = []
                            for eid in raw_ids:
                                # match against 'unique_app_id' or 'id' column
                                mask = pd.Series(False, index=df_analysis.index)
                                if 'unique_app_id' in df_analysis.columns:
                                    mask = df_analysis['unique_app_id'].astype(str) == eid
                                elif 'id' in df_analysis.columns:
                                    mask = df_analysis['id'].astype(str) == eid
                                matched = df_analysis[mask]
                                if not matched.empty:
                                    example_texts.append(matched.iloc[0].get('text', ''))
                            code_metadata_list.append({
                                "code_id": str(row.get("code_id", "")),
                                "code_name": row.get("Code Name", ""),
                                "description": row.get("Description", ""),
                                "rationale": row.get("Rationale", ""),
                                "example_texts": example_texts
                            })
                        else:
                            code_metadata_list.append({
                                "code_id": "",
                                "code_name": code,
                                "description": "",
                                "rationale": "",
                                "example_texts": []
                            })
                    unassigned_codes_json_str = json.dumps(code_metadata_list, indent=2)
                    with st.spinner(f"Sending {len(unassigned_codes_for_ai)} codes to AI..."):
                        prompt_with_groups = st.session_state.ai_grouping_prompt_template_p03.replace("{existing_groups_json}", existing_groups_json_str)
                        ai_response_groups = ai_services.generate_code_groups_with_ai(unassigned_codes_json_str, prompt_with_groups, ai_provider_analysis, ai_model_analysis)
                    # Apply AI-suggested groups immediately, update table, then show JSON modal
                    if isinstance(ai_response_groups, list) and all(isinstance(g, dict) and "group_name" in g and "codes" in g for g in ai_response_groups):
                        # Build fresh groups dict
                        new_groups = {}
                        for group_data in ai_response_groups:
                            ai_g_name = str(group_data["group_name"]).strip()
                            raw_codes = group_data.get("codes", []) or []
                            cleaned_codes = [str(c).replace("\\/", "/").strip() for c in raw_codes if str(c).strip()]
                            if ai_g_name and cleaned_codes:
                                new_groups[ai_g_name] = cleaned_codes
                        # Update in-memory groups and table
                        st.session_state.created_code_groups = new_groups
                        update_dataframe_groups_column()
                        # Now show the raw JSON for inspection
                        ai_group_json_dialog()
                    else:
                        err_d = str(ai_response_groups) if ai_response_groups else "No response."
                        ui_helpers.show_error_message(f"AI failed. Details: {err_d[:200]}...")
                        logger.error(f"AI group failed: {ai_response_groups}")
    
    with tab_manual_edit_group:
        st.header("Manage Group Assignments")
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
                    selected_codes_create_new_form = [] 
                    with st.container(height=300):
                        for idx_code, code in enumerate(codes_to_display_checkboxes):
                            col_cb, col_btn = st.columns([0.75, 0.25])
                            checkbox_key = f"cb_new_uncat_{utils.generate_project_id(code)}_{idx_code}"
                            is_checked = col_cb.checkbox(
                                code,
                                value=st.session_state.manual_edit_codes_checkboxes_state.get(code, False),
                                key=checkbox_key
                            )
                            if st.session_state.manual_edit_codes_checkboxes_state.get(code, False) != is_checked:
                                st.session_state.manual_edit_codes_checkboxes_state[code] = is_checked

                            if is_checked: 
                                selected_codes_create_new_form.append(code)

                            if col_btn.button("Inspect", key=f"inspect_btn_uncat_new_{utils.generate_project_id(code)}_{idx_code}"):
                                df_all = st.session_state.analysis_table_df.copy()
                                details_df = df_all[df_all['Codes'].apply(lambda s: code in robust_comma_string_to_list_analysis(s))]
                                st.session_state.inspect_dialog_code_occurrences = details_df
                                st.session_state.inspect_dialog_code_name = code
                                inspect_code_occurrences_dialog(title=f"Occurrences of '{code}'")

                    if st.button("Create Group", key="create_group_manual_btn_col2_form"):
                        new_g_name = st.session_state.manual_edit_new_group_name_input.strip()
                        codes_for_new_group = selected_codes_create_new_form
                        if not new_g_name:
                            ui_helpers.show_error_message("Name empty.")
                        elif not codes_for_new_group: 
                            ui_helpers.show_error_message("Select codes.")
                        elif new_g_name in st.session_state.created_code_groups:
                            ui_helpers.show_error_message(f"Group '{new_g_name}' exists.")
                        else:
                            st.session_state.created_code_groups[new_g_name] = codes_for_new_group.copy()
                            ui_helpers.show_success_message(f"Group '{new_g_name}' created.")
                            st.session_state.manual_edit_new_group_name_input = ""
                            st.session_state.manual_edit_codes_checkboxes_state = {} 
                            st.rerun()
                
                elif st.session_state.manual_edit_uncategorised_action_radio == "Add to Existing Group":
                    existing_g_names = sorted(list(st.session_state.created_code_groups.keys()))
                    if not existing_g_names: st.warning("No existing groups. Create one first.")
                    else:
                        with st.form("col2_add_to_existing_form"):
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
                                    ui_helpers.show_success_message(f"Codes added to '{target_g}'."); 
                                    st.session_state.manual_edit_codes_checkboxes_state = {}; 
                                    st.session_state.manual_edit_selected_group_radio = target_g; 
                                    st.rerun()
            
            elif current_selection_context in st.session_state.created_code_groups: 
                st.session_state.manual_edit_code_search_input = st.text_input(f"Search codes within '{current_selection_context}':", value=st.session_state.manual_edit_code_search_input, key="col2_search_in_selected_group")
                codes_in_selected_group = sorted(st.session_state.created_code_groups[current_selection_context])
                codes_to_display_checkboxes = [c for c in codes_in_selected_group if st.session_state.manual_edit_code_search_input.lower() in c.lower()] if st.session_state.manual_edit_code_search_input else codes_in_selected_group
                
                st.caption(f"Codes in '{current_selection_context}'. Uncheck to remove from this group upon update.")
                with st.container(height=200): 
                    for code in codes_to_display_checkboxes: 
                        cb_key = f"cb_existing_g_{utils.generate_project_id(code)}_{utils.generate_project_id(current_selection_context)}_col2"
                        col_cb_exist, col_btn_exist = st.columns([0.8, 0.2])
                        with col_cb_exist:
                            is_checked_now = st.checkbox(code, value=st.session_state.manual_edit_codes_checkboxes_state.get(code, True), 
                                        key=cb_key, on_change=handle_checkbox_change, args=(code, cb_key))
                        with col_btn_exist:
                            if st.button("Inspect", key=f"inspect_btn_exist_g_{utils.generate_project_id(code)}_{utils.generate_project_id(current_selection_context)}"):
                                df_all = st.session_state.analysis_table_df.copy()
                                details_df = df_all[df_all['Codes'].apply(lambda s: code in robust_comma_string_to_list_analysis(s))]
                                st.session_state.inspect_dialog_code_occurrences = details_df
                                st.session_state.inspect_dialog_code_name = code
                                inspect_code_occurrences_dialog(title=f"Occurrences of '{code}'") 

                if st.button(f"Update Group '{current_selection_context}' (Remove Unchecked)", key="col2_update_group_btn"):
                    codes_to_keep = [c for c in codes_in_selected_group if st.session_state.manual_edit_codes_checkboxes_state.get(c, True)] 
                    original_codes_this_group = set(st.session_state.created_code_groups[current_selection_context]); removed_count = len(original_codes_this_group - set(codes_to_keep))
                    if not codes_to_keep: del st.session_state.created_code_groups[current_selection_context]; ui_helpers.show_success_message(f"Group '{current_selection_context}' removed."); st.session_state.manual_edit_selected_group_radio = "Uncategorised Codes"
                    else: st.session_state.created_code_groups[current_selection_context] = codes_to_keep; ui_helpers.show_success_message(f"Group '{current_selection_context}' updated. {removed_count} code(s) uncategorised.")
                    st.session_state.manual_edit_codes_checkboxes_state = {}; st.rerun() 
            else: st.empty()
else: 
    st.info("Select one or more views to load data for analysis and grouping.")

# --- Dialog Definition (moved up to ensure it's defined before use) ---

@st.dialog("Inspect Code Occurrences")
def inspect_code_occurrences_dialog(title="Inspect Code Occurrences"):
    st.title(title)
    occurrences_df = st.session_state.get("inspect_dialog_code_occurrences", pd.DataFrame())
    if not occurrences_df.empty:
        cols_to_show = [col for col in ['unique_app_id', 'id', 'text', 'title', 'Codes', 'groups'] if col in occurrences_df.columns]
        st.dataframe(occurrences_df[cols_to_show], use_container_width=True)
    else:
        st.info(f"No occurrences found for code: {st.session_state.get('inspect_dialog_code_name', 'N/A')}")
    if st.button("Close Inspection Dialog", key="close_inspect_code_occurrences_dialog_btn"): 
        if "inspect_dialog_code_occurrences" in st.session_state:
            del st.session_state.inspect_dialog_code_occurrences
        if "inspect_dialog_code_name" in st.session_state:
            del st.session_state.inspect_dialog_code_name
        st.rerun()

