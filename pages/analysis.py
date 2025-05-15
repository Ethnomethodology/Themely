# pages/analysis.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from modules import data_manager, ui_helpers, utils, ai_services
import json
import re # Import re for regex in group filtering
import math # For ceiling function for sample size

logger = utils.setup_logger("p04_analysis")

# --- Constants ---
ANALYSIS_SUBFOLDER = "Analysis"
SUMMARIES_FILENAME = "group_summaries.csv"
DEFAULT_SUMMARY_PROMPT = """You are an expert qualitative research analyst specializing in synthesizing thematic data.

CONTEXT:
You will be given a thematic "Group Name" and a JSON array of "Text Samples" belonging to this group.
Group Name: {GROUP_NAME}
Text Samples (JSON Array):
{TEXT_SAMPLES_JSON}

TASK:
Your task is to write a concise, objective summary that captures the core essence, main ideas, and any significant nuances of this thematic group. This summary should be based *only* on the provided text samples.
Target length is 2-4 sentences, with a maximum of 5 sentences.

RULES:
1.  The summary MUST be strictly grounded in the provided "Text Samples". Do not infer, invent, or add any external information or common knowledge.
2.  Focus on identifying and reflecting the dominant patterns, key insights, and prevalent sentiments within the texts.
3.  Maintain a neutral and objective tone throughout the summary.
4.  The summary should be presented as a single, coherent paragraph.
5.  Ensure the summary is easy to read and understand.

OUTPUT FORMAT:
Return ONLY the summary as a plain text string.
Do NOT include:
    - Any introductory or concluding phrases (e.g., "Here is the summary:", "In conclusion,").
    - Any mention of the group name within the summary text itself unless it's naturally part of the synthesized information from the texts.
    - Any markdown formatting (like headers, bolding, lists).
    - Any JSON structure, code fences, or any characters outside the plain text summary.
"""

# --- Initialize Session State Variables for this Page ---
if 'analysis_selected_views_info_p04' not in st.session_state:
    st.session_state.analysis_selected_views_info_p04 = {}
if 'analysis_group_summaries_p04' not in st.session_state:
    st.session_state.analysis_group_summaries_p04 = {}
if 'analysis_active_tab_group_p04' not in st.session_state:
    st.session_state.analysis_active_tab_group_p04 = None

# Modal/Dialog States
if 'analysis_show_summary_modal_p04' not in st.session_state:
    st.session_state.analysis_show_summary_modal_p04 = False
if 'analysis_current_group_for_summary_p04' not in st.session_state:
    st.session_state.analysis_current_group_for_summary_p04 = None
if 'analysis_summary_text_col_p04' not in st.session_state:
    st.session_state.analysis_summary_text_col_p04 = "text"
if 'analysis_summary_prompt_template_p04' not in st.session_state:
    st.session_state.analysis_summary_prompt_template_p04 = DEFAULT_SUMMARY_PROMPT
if 'analysis_ai_model_summary_p04' not in st.session_state: 
    st.session_state.analysis_ai_model_summary_p04 = None
if 'analysis_ai_model_summary_p04_openai' not in st.session_state: 
    st.session_state.analysis_ai_model_summary_p04_openai = "gpt-4o" # Default OpenAI model
if 'analysis_ai_model_summary_p04_gemini' not in st.session_state:
    st.session_state.analysis_ai_model_summary_p04_gemini = "gemini-2.0-flash" # Default Gemini model
if 'analysis_sample_percentage_p04' not in st.session_state:
    st.session_state.analysis_sample_percentage_p04 = 10

if 'analysis_newly_generated_summary_p04' not in st.session_state:
    st.session_state.analysis_newly_generated_summary_p04 = None
if 'analysis_ai_raw_summary_json_p04' not in st.session_state:
    st.session_state.analysis_ai_raw_summary_json_p04 = ""
if 'analysis_show_ai_raw_summary_modal_p04' not in st.session_state:
    st.session_state.analysis_show_ai_raw_summary_modal_p04 = False

if 'analysis_combined_df_p04' not in st.session_state:
    st.session_state.analysis_combined_df_p04 = pd.DataFrame()

# --- Helper Functions for Data Persistence ---
def get_summaries_filepath(project_path):
    if not project_path: return None
    analysis_dir = os.path.join(project_path, ANALYSIS_SUBFOLDER)
    return os.path.join(analysis_dir, SUMMARIES_FILENAME)

def load_group_summaries_from_file():
    if not st.session_state.get("project_path"):
        logger.warning("Project path not set, cannot load summaries.")
        st.session_state.analysis_group_summaries_p04 = {}
        return

    summaries_path = get_summaries_filepath(st.session_state.project_path)
    if summaries_path and os.path.exists(summaries_path):
        try:
            df_summaries = pd.read_csv(summaries_path)
            loaded_summaries = {str(row['group_name']): str(row['summary']) for _, row in df_summaries.iterrows()}
            st.session_state.analysis_group_summaries_p04 = loaded_summaries
            logger.info(f"Group summaries loaded from {summaries_path}")
        except Exception as e:
            logger.error(f"Error loading group summaries from {summaries_path}: {e}")
            ui_helpers.show_error_message(f"Could not load existing summaries: {e}")
            st.session_state.analysis_group_summaries_p04 = {}
    else:
        logger.info("No existing group summaries file found.")
        st.session_state.analysis_group_summaries_p04 = {}

def save_group_summaries_to_file():
    if not st.session_state.get("project_path"):
        ui_helpers.show_error_message("Project path not set. Cannot save summaries.")
        return False

    analysis_dir = os.path.join(st.session_state.project_path, ANALYSIS_SUBFOLDER)
    try:
        os.makedirs(analysis_dir, exist_ok=True)
    except OSError as e:
        ui_helpers.show_error_message(f"Could not create Analysis directory '{analysis_dir}': {e}")
        return False

    summaries_path = get_summaries_filepath(st.session_state.project_path)
    summaries_to_save = st.session_state.analysis_group_summaries_p04
    
    df_data = []
    for group_name, summary in summaries_to_save.items():
        df_data.append({'group_name': str(group_name), 'summary': str(summary)})
    df_summaries = pd.DataFrame(df_data)

    if df_summaries.empty and not summaries_to_save:
         df_summaries = pd.DataFrame(columns=['group_name', 'summary'])

    try:
        df_summaries.to_csv(summaries_path, index=False)
        logger.info(f"Group summaries saved to {summaries_path}")
        ui_helpers.show_success_message("Group summaries saved successfully.")
        return True
    except Exception as e:
        logger.error(f"Error saving group summaries to {summaries_path}: {e}")
        ui_helpers.show_error_message(f"Failed to save summaries: {e}")
        return False

# --- Dialog for AI Summary Generation Settings ---
@st.dialog("Generate Group Summary with AI", width="large")
def summary_generation_settings_dialog():
    group_name = st.session_state.analysis_current_group_for_summary_p04
    st.title(f"AI Summary for Group: {group_name}")

    combined_df = st.session_state.analysis_combined_df_p04
    if combined_df.empty:
        st.error("No data loaded for summary generation.")
        if st.button("Close", key="sum_diag_close_nodata"): st.session_state.analysis_show_summary_modal_p04 = False; st.rerun()
        return

    group_df = combined_df[combined_df['groups'].astype(str).str.contains(f'(?<![^, ]){re.escape(str(group_name))}(?![^, ])', na=False, regex=True)]
    if group_df.empty:
        st.error(f"No data found for group '{group_name}' in the currently loaded views.")
        if st.button("Close", key="sum_diag_close_nogrpdata"): st.session_state.analysis_show_summary_modal_p04 = False; st.rerun()
        return

    text_cols = [col for col in group_df.columns if group_df[col].dtype == 'object' and col not in ['groups', 'Codes', 'unique_app_id', 'id', 'Source View']]
    if not text_cols:
        st.error("No suitable text columns found in the data for this group.")
        if st.button("Close", key="sum_diag_close_notxtcol"): st.session_state.analysis_show_summary_modal_p04 = False; st.rerun()
        return

    current_text_col_val = st.session_state.analysis_summary_text_col_p04
    try:
        current_text_col_idx = text_cols.index(current_text_col_val) if current_text_col_val in text_cols else (text_cols.index("text") if "text" in text_cols else 0)
    except ValueError:
        current_text_col_idx = 0
    st.session_state.analysis_summary_text_col_p04 = text_cols[current_text_col_idx]

    st.session_state.analysis_summary_text_col_p04 = st.selectbox(
        "Select Text Column for Summary Input:",
        text_cols,
        index=current_text_col_idx,
        key="sum_diag_text_col_select"
    )

    st.session_state.analysis_summary_prompt_template_p04 = st.text_area(
        "AI Summary Prompt Template:",
        value=st.session_state.analysis_summary_prompt_template_p04,
        height=330,
        key="sum_diag_prompt_template_input"
    )

    ai_provider = st.session_state.project_config.get('ai_provider', "OpenAI")
    st.info(f"Using AI Provider: **{ai_provider}**")

    openai_models = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    gemini_models = ["gemini-2.0-flash", "gemini-1.5-flash"] # Corrected and aligned

    if ai_provider == "OpenAI":
        current_model = st.session_state.get('analysis_ai_model_summary_p04_openai', "gpt-4o")
        if current_model not in openai_models: current_model = "gpt-4o"
        selected_model_openai = st.selectbox("OpenAI Model:", openai_models, index=openai_models.index(current_model), key="sum_diag_openai_model")
        st.session_state.analysis_ai_model_summary_p04_openai = selected_model_openai
        st.session_state.analysis_ai_model_summary_p04 = selected_model_openai
    elif ai_provider == "Gemini":
        current_model = st.session_state.get('analysis_ai_model_summary_p04_gemini', "gemini-2.0-flash")
        if current_model not in gemini_models: current_model = "gemini-2.0-flash"
        selected_model_gemini = st.selectbox("Gemini Model:", gemini_models, index=gemini_models.index(current_model), key="sum_diag_gemini_model")
        st.session_state.analysis_ai_model_summary_p04_gemini = selected_model_gemini
        st.session_state.analysis_ai_model_summary_p04 = selected_model_gemini
    else:
        st.session_state.analysis_ai_model_summary_p04 = None
    
    st.session_state.analysis_sample_percentage_p04 = st.slider(
        "Percentage of text samples to send to AI:",
        min_value=1, max_value=100, value=st.session_state.analysis_sample_percentage_p04, step=1,
        key="sum_diag_sample_percentage_slider"
    )
    
    texts_for_summary_series = group_df[st.session_state.analysis_summary_text_col_p04].dropna()
    if not texts_for_summary_series.empty:
        actual_sample_count = math.ceil(len(texts_for_summary_series) * (st.session_state.analysis_sample_percentage_p04 / 100.0))
        actual_sample_count = max(1, actual_sample_count) 
        st.caption(f"This will send {actual_sample_count} text sample(s) to the AI (out of {len(texts_for_summary_series)} available for this group and column).")
    else:
        actual_sample_count = 0
        st.caption("No text data available in the selected column for this group to sample.")

    if st.button("Generate Summary", key="sum_diag_generate_btn", disabled=(not st.session_state.analysis_ai_model_summary_p04 or actual_sample_count == 0)):
        if "{GROUP_NAME}" not in st.session_state.analysis_summary_prompt_template_p04 or \
           "{TEXT_SAMPLES_JSON}" not in st.session_state.analysis_summary_prompt_template_p04:
            ui_helpers.show_error_message("Prompt template is missing required placeholders: {GROUP_NAME} and/or {TEXT_SAMPLES_JSON}.")
            return

        if texts_for_summary_series.empty:
            ui_helpers.show_error_message(f"No text data available in column '{st.session_state.analysis_summary_text_col_p04}' for group '{group_name}'.")
            return
        
        texts_for_summary = texts_for_summary_series.sample(min(actual_sample_count, len(texts_for_summary_series))).tolist()
        text_samples_json_str = json.dumps(texts_for_summary, indent=2)

        with st.spinner(f"Generating summary for '{group_name}' with {ai_provider}..."):
            summary_text, raw_response = ai_services.generate_summary_for_group_with_ai(
                group_name=str(group_name),
                text_samples_json=text_samples_json_str,
                prompt_template=st.session_state.analysis_summary_prompt_template_p04,
                provider_name=ai_provider,
                model_name=st.session_state.analysis_ai_model_summary_p04
            )
        
        st.session_state.analysis_ai_raw_summary_json_p04 = raw_response
        if isinstance(summary_text, str) and not summary_text.startswith("Error:"):
            st.session_state.analysis_newly_generated_summary_p04 = summary_text
            ui_helpers.show_success_message(f"AI summary generated for '{group_name}'.")
        else:
            ui_helpers.show_error_message(summary_text)
            st.session_state.analysis_newly_generated_summary_p04 = summary_text

        st.session_state.analysis_show_summary_modal_p04 = False
        st.session_state.analysis_show_ai_raw_summary_modal_p04 = True
        st.rerun()

    if st.button("Cancel", key="sum_diag_cancel_btn"):
        st.session_state.analysis_show_summary_modal_p04 = False
        st.rerun()

# --- Dialog for AI Raw Summary JSON ---
@st.dialog("AI Raw Summary JSON", width="large")
def ai_raw_summary_json_dialog():
    active_group = st.session_state.get("analysis_current_group_for_summary_p04", "N/A")
    st.title(f"AI Raw Response for Group: {active_group}")
    raw_response_str = st.session_state.get("analysis_ai_raw_summary_json_p04", "")
    if raw_response_str:
        try:
            parsed_json = json.loads(raw_response_str)
            st.code(json.dumps(parsed_json, indent=2), language="json")
        except json.JSONDecodeError:
            st.code(raw_response_str, language="text")
    else:
        st.info("No AI JSON response available.")

    if st.button("Close", key="close_ai_raw_summary_json_dialog"):
        st.session_state.analysis_show_ai_raw_summary_modal_p04 = False
        if active_group != "N/A" and st.session_state.analysis_newly_generated_summary_p04 is not None:
            summary_key_to_update = f"summary_edit_area_{str(active_group).replace(' ', '_')}_p04"
            st.session_state[summary_key_to_update] = st.session_state.analysis_newly_generated_summary_p04
        st.rerun()

# --- Main Page Logic ---
st.title("Analysis")

if not st.session_state.get('current_project_name'):
    st.warning("👈 Please create or open a project first from the '🏠 Project Setup' page.")
    st.stop()

# --- 1. View Selection ---
st.subheader("Select Grouped View(s) for Analysis")
available_views_meta_analysis = data_manager.list_created_views_metadata()

if not available_views_meta_analysis:
    st.info("No project views created yet. Go to '💾 Data Management' to create views, and 'Themes' to group the codes.")
else:
    for vm in available_views_meta_analysis:
        view_key = vm["view_name"] + "_analysis_select_p04"
        if view_key not in st.session_state.analysis_selected_views_info_p04:
            st.session_state.analysis_selected_views_info_p04[view_key] = {"selected": False, "metadata": vm}

    views_selection_df_data = []
    for vm in available_views_meta_analysis:
        views_selection_df_data.append({
            "View Name": vm["view_name"],
            "Created On": datetime.fromisoformat(vm.get("creation_timestamp", "")).strftime("%Y-%m-%d %H:%M") if vm.get("creation_timestamp") else "N/A",
            "Source Files Info": ", ".join(vm.get("source_files_info") if isinstance(vm.get("source_files_info"), list) else [str(vm.get("source_files_info", ""))])
        })
    views_selection_df = pd.DataFrame(views_selection_df_data)

    view_selection_event = st.dataframe(
        views_selection_df,
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="multi-row",
        key="analysis_view_selector_df_p04"
    )

    selected_idxs = view_selection_event.selection.rows if hasattr(view_selection_event, 'selection') else []
    selection_changed = False
    temp_selected_view_names_on_interaction = []

    for idx, vm_meta in enumerate(available_views_meta_analysis):
        key = vm_meta["view_name"] + "_analysis_select_p04"
        current_selection_state = st.session_state.analysis_selected_views_info_p04.get(key, {}).get("selected", False)
        new_selection_state = idx in selected_idxs

        if current_selection_state != new_selection_state:
            selection_changed = True
        st.session_state.analysis_selected_views_info_p04[key]["selected"] = new_selection_state
        if new_selection_state:
            temp_selected_view_names_on_interaction.append(vm_meta["view_name"])

    if selection_changed:
        if temp_selected_view_names_on_interaction:
            combined_df_list = []
            for view_name in temp_selected_view_names_on_interaction:
                view_info_dict_entry = st.session_state.analysis_selected_views_info_p04.get(view_name + "_analysis_select_p04")
                if view_info_dict_entry and view_info_dict_entry["selected"]:
                    view_meta = view_info_dict_entry["metadata"]
                    view_path = view_meta.get("csv_filepath")
                    if view_path and os.path.exists(view_path):
                        df = data_manager.load_data_from_specific_file(view_path)
                        if df is not None:
                            combined_df_list.append(df)
                    else:
                        st.warning(f"CSV file for view '{view_name}' not found at {view_path}.")
            if combined_df_list:
                dedup_col = None
                first_valid_df = next((df for df in combined_df_list if df is not None and not df.empty), None)
                if first_valid_df is not None:
                    if 'unique_app_id' in first_valid_df.columns: dedup_col = 'unique_app_id'
                    elif 'id' in first_valid_df.columns: dedup_col = 'id'
                
                concat_df = pd.concat(combined_df_list, ignore_index=True)
                if dedup_col and dedup_col in concat_df.columns:
                    st.session_state.analysis_combined_df_p04 = concat_df.drop_duplicates(subset=[dedup_col], keep='first')
                else:
                    st.session_state.analysis_combined_df_p04 = concat_df
            else:
                st.session_state.analysis_combined_df_p04 = pd.DataFrame()
        else:
            st.session_state.analysis_combined_df_p04 = pd.DataFrame()

        load_group_summaries_from_file()
        st.session_state.analysis_newly_generated_summary_p04 = None
        st.session_state.analysis_current_group_for_summary_p04 = None
        st.rerun()

    current_combined_df = st.session_state.analysis_combined_df_p04
    selected_view_names_for_display = [
        vm_meta["view_name"] for vm_meta in available_views_meta_analysis
        if st.session_state.analysis_selected_views_info_p04.get(vm_meta["view_name"] + "_analysis_select_p04", {}).get("selected")
    ]

    if not current_combined_df.empty:
        if 'groups' not in current_combined_df.columns or current_combined_df['groups'].astype(str).str.strip().eq('').all():
            st.warning("The selected view(s) do not contain a 'groups' column with actual group data. Please ensure codes are grouped on the 'Themes' page first.")
        else:
            all_groups_list = []
            for groups_str in current_combined_df['groups'].dropna().astype(str):
                if groups_str.strip():
                    all_groups_list.extend([g.strip() for g in groups_str.split(',') if g.strip()])
            
            unique_groups = sorted(list(set(str(g) for g in all_groups_list if g)))

            if unique_groups:
                if st.session_state.analysis_active_tab_group_p04 not in unique_groups:
                    st.session_state.analysis_active_tab_group_p04 = unique_groups[0] if unique_groups else None
                
                group_tabs = st.tabs(unique_groups)

                for i, group_name_tab_str in enumerate(unique_groups):
                    with group_tabs[i]:
                        st.subheader(f"Analysis for Group: {group_name_tab_str}")

                        group_df_display = current_combined_df[current_combined_df['groups'].astype(str).str.contains(f'(?<![^, ]){re.escape(group_name_tab_str)}(?![^, ])', na=False, regex=True)]
                        if not group_df_display.empty:
                            st.dataframe(group_df_display, use_container_width=True, height=300)
                        else:
                            st.info(f"No data items found for group '{group_name_tab_str}' in the selected views.")

                        st.markdown("---")
                        st.subheader("Group Summary")

                        summary_key_for_group = f"summary_edit_area_{group_name_tab_str.replace(' ', '_')}_p04"

                        if summary_key_for_group not in st.session_state:
                            st.session_state[summary_key_for_group] = st.session_state.analysis_group_summaries_p04.get(group_name_tab_str, "")

                        if st.session_state.analysis_current_group_for_summary_p04 == group_name_tab_str and \
                           st.session_state.analysis_newly_generated_summary_p04 is not None:
                            st.session_state[summary_key_for_group] = st.session_state.analysis_newly_generated_summary_p04
                        
                        st.text_area(
                            "Summary (edit and save):",
                            key=summary_key_for_group, 
                            height=150
                        )

                        col1, col2, col3, col4_spacer = st.columns([1,1,1,2])
                        with col1:
                            if st.button("Generate/Regenerate Summary", key=f"generate_summary_btn_{group_name_tab_str.replace(' ', '_')}_p04", use_container_width=True):
                                st.session_state.analysis_current_group_for_summary_p04 = group_name_tab_str
                                st.session_state.analysis_show_summary_modal_p04 = True
                                st.session_state.analysis_newly_generated_summary_p04 = None
                                st.rerun()
                        with col2:
                            if st.button("Save Summary", key=f"save_summary_btn_{group_name_tab_str.replace(' ', '_')}_p04", use_container_width=True):
                                st.session_state.analysis_group_summaries_p04[group_name_tab_str] = st.session_state[summary_key_for_group]
                                save_group_summaries_to_file()
                                if st.session_state.analysis_current_group_for_summary_p04 == group_name_tab_str:
                                    st.session_state.analysis_newly_generated_summary_p04 = None
                                st.rerun()
                        with col3:
                            if st.button("Discard Changes", key=f"discard_summary_btn_{group_name_tab_str.replace(' ', '_')}_p04", use_container_width=True): # Corrected f-string
                                st.session_state[summary_key_for_group] = st.session_state.analysis_group_summaries_p04.get(group_name_tab_str, "")
                                if st.session_state.analysis_current_group_for_summary_p04 == group_name_tab_str:
                                     st.session_state.analysis_newly_generated_summary_p04 = None
                                st.rerun()
            else:
                st.info("No groups found in the 'groups' column of the selected views. Please group codes on the 'Themes' page.")
    elif selected_view_names_for_display:
        st.info("No data to display from the selected views. They might be empty or failed to load.")
    else:
        st.info("👈 Select one or more views from the table above to begin analysis.")

if st.session_state.analysis_show_summary_modal_p04:
    summary_generation_settings_dialog()

if st.session_state.analysis_show_ai_raw_summary_modal_p04:
    ai_raw_summary_json_dialog()