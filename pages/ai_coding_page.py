# pages/ai_coding_page.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from modules import data_management, ui_helpers, utils, ai_services
import math # For sqrt in codebook generation prompt info

logger = utils.setup_logger("p02_ai_coding")

# Dialog for editing a single code entry
@st.dialog("Edit Code", width="large")
def edit_code_dialog():
    idx = st.session_state.get("edit_dialog_selected_index", None)
    if idx is None:
        st.error("No code selected for editing.")
        if st.button("Close"):
            st.rerun()
        return
    # Fetch the row data
    row = st.session_state.edited_codebook_df.loc[idx]
    # Editable fields
    name = st.text_input("Code Name*", value=row["Code Name"], key="edit_code_name")
    desc = st.text_area("Description", value=row["Description"], key="edit_code_description")
    rationale = st.text_area("Rationale", value=row["Rationale"], key="edit_code_rationale")
    ex_ids = st.text_area("Example IDs (comma-separated)", value=row.get("Example_ids", ""), key="edit_code_example_ids")
    # Submit changes
    if st.button("Submit"):
        df = st.session_state.edited_codebook_df.copy()
        df.at[idx, "Code Name"] = name.strip()
        df.at[idx, "Description"] = desc.strip()
        df.at[idx, "Rationale"] = rationale.strip()
        df.at[idx, "Example_ids"] = ex_ids.strip()
        st.session_state.edited_codebook_df = df
        st.rerun()

# --- Dialog for Inspecting Codebook Examples ---
@st.dialog("Inspect Codebook Examples", width="large")
def inspect_codebook_examples_dialog():
    selected_code_names = st.session_state.get("inspect_dialog_selected_code_names", [])
    example_ids_str_combined = st.session_state.get("inspect_dialog_example_ids_str_combined", "")
    id_column_for_examples = st.session_state.get("inspect_dialog_id_column", None)

    if not selected_code_names:
        st.subheader("No codes were selected for inspection.")
    else:
        st.subheader(f"Examples for Code(s): {', '.join(selected_code_names)}")

    if not example_ids_str_combined:
        st.write("No example IDs found for the selected code(s).")
        if st.button("Close"):
            st.rerun()
        return

    example_ids_list = [eid.strip() for eid in example_ids_str_combined.split(',') if eid.strip()]
    if not example_ids_list:
        st.write("No valid example IDs found after parsing.")
        if st.button("Close"):
            st.rerun()
        return

    data_to_filter = st.session_state.get('data_for_coding_tab3', pd.DataFrame())

    if data_to_filter.empty:
        st.warning("Source data for examples ('Data to be Coded') is not loaded.")
        if st.button("Close"):
            st.rerun()
        return

    actual_id_col = None
    if id_column_for_examples and id_column_for_examples in data_to_filter.columns:
        actual_id_col = id_column_for_examples
    elif 'unique_app_id' in data_to_filter.columns:
        actual_id_col = 'unique_app_id'
    elif 'id' in data_to_filter.columns:
        actual_id_col = 'id'
    
    if not actual_id_col:
        st.error("Could not determine a valid ID column in the source data to match Example IDs ('unique_app_id' or 'id' not found).")
        if st.button("Close"):
            st.rerun()
        return
    
    st.caption(f"Matching Example IDs against column: `{actual_id_col}` in the 'Data to be Coded' table.")

    try:
        example_ids_list_str = list(set([str(eid) for eid in example_ids_list]))
        filtered_df = data_to_filter[data_to_filter[actual_id_col].astype(str).isin(example_ids_list_str)]
    except Exception as e:
        st.error(f"Error filtering data for examples: {e}")
        if st.button("Close"):
            st.rerun()
        return

    if filtered_df.empty:
        st.write(f"No items found in 'Data to be Coded' matching the Example IDs: {', '.join(example_ids_list_str)}")
    else:
        st.dataframe(filtered_df, use_container_width=True)

    if st.button("Close"):
        st.rerun()


# --- Dialog for Updating Codes on a Single Row ---
@st.dialog("Update Codes", width="medium")
def update_codes_dialog():
    # Use the single selected row index from section 2
    idx_list = st.session_state.get("selected_section2_indices", [])
    idx = idx_list[0] if len(idx_list) == 1 else None
    if idx is None:
        st.error("No row selected for updating codes.")
        if st.button("Close"):
            st.rerun()
        return
    # Fetch current codes for this row
    row = st.session_state.data_for_coding_tab3.loc[idx]
    current_codes = robust_comma_string_to_list_p02(row.get("Codes", ""))
    # Available codes come from the codebook draft
    options = st.session_state.edited_codebook_df["Code Name"].tolist()
    selected_codes = st.multiselect("Select Codes", options=options, default=current_codes, key="update_codes_multiselect")
    if st.button("Save"):
        # Update in-memory
        new_codes_str = robust_list_to_comma_string_p02(selected_codes)
        st.session_state.data_for_coding_tab3.at[idx, "Codes"] = new_codes_str
        # Persist to underlying file if exactly one view is loaded
        selected_paths = st.session_state.get("currently_selected_view_filepaths_for_saving_tab3", [])
        if len(selected_paths) == 1:
            df_to_save = st.session_state.data_for_coding_tab3.copy()
            if "Source View" in df_to_save.columns:
                df_to_save = df_to_save.drop(columns=["Source View"], errors="ignore")
            success = data_management.save_coded_data_to_view(df_to_save, selected_paths[0])
            if success:
                ui_helpers.show_success_message(f"Codes updated in '{os.path.basename(selected_paths[0])}'.")
            else:
                ui_helpers.show_error_message(f"Failed to update codes in '{os.path.basename(selected_paths[0])}'.")
        else:
            ui_helpers.show_info("Codes updated in memory. Use 'Save All Row Codes' to persist.")
        st.rerun()

# --- Dialog for Asking AI to Code All Rows (Section 2) ---
@st.dialog("Ask AI to Code", width="medium")
def ask_ai_code_dialog():
    # Prepare inputs
    df_data = st.session_state.data_for_coding_tab3.copy()
    # Codebook JSON array (for injection, not for default template)
    codebook_df = st.session_state.edited_codebook_df.drop(columns=["Select"], errors="ignore").reset_index(drop=True)
    codebook_json = codebook_df.to_json(orient="records")
    # Column selectors
    text_columns = [col for col in df_data.columns if df_data[col].dtype == "object" and col not in ["Source View", "Codes"]]
    id_columns = [col for col in ["unique_app_id", "id"] if col in df_data.columns]
    selected_text_col = st.selectbox("Select Text Column:", text_columns, index=0, key="ask_ai_code_text_col")
    selected_id_col = st.selectbox("Select ID Column:", id_columns, index=0, key="ask_ai_code_id_col")
    # AI model selector
    ai_provider = st.session_state.project_config.get("ai_provider", "OpenAI")
    ai_model = None
    if ai_provider == "OpenAI":
        ai_model = st.selectbox(
            "OpenAI Model:",
            ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0,
            key="ask_ai_code_model"
        )
    else:
        # Gemini model selector
        if ai_provider == "Gemini":
            ai_model = st.selectbox(
                f"{ai_provider} Model:",
                ["gemini-2.0-flash", "gemini-1.5-flash"],
                index=0,
                key="ask_ai_code_model"
            )
        else:
            ai_model = st.selectbox(
                f"{ai_provider} Model:",
                [],
                key="ask_ai_code_model"
            )

    # Editable prompt template for AI coding (show placeholders, not actual data)
    default_template = """You are an experienced qualitative researcher applying an existing codebook on Reddit data.

CODEBOOK
{CODEBOOK_JSON_ARRAY}

TASK
For every item in the input array:
1. Read its "{TEXT_COLUMN_NAME}" field.
2. Decide which Code labels from the codebook apply.
   • Zero, one, or multiple codes may apply (cap at 5 per item).  
3. Output an object with exactly two keys:
   • "{ID_COLUMN_NAME}" — the original identifier (string)  
   • "Codes" — a single comma-separated string of the applicable Code labels,
               or an empty string if none fit.

RULES
• Use Code labels **exactly** as written in the codebook.  
• Do **not** invent, rename, or combine codes.  
• Alphabetise the Codes string for consistency.  
• No explanations, markdown, or extra keys.

OUTPUT  
Return **only** a valid JSON array of these row objects. Do not include any introductory text, explanations, markdown formatting, or any characters outside the main JSON array structure.

INPUT DATA
{JSON_DATA_BATCH}
"""
    prompt_to_use = st.text_area("AI Coding Prompt Template:", value=default_template, height=300, key="ask_ai_code_prompt_template_input")

    # Batch size input for chunked AI calls
    batch_size = st.number_input(
        "Batch Size",
        min_value=1,
        value=10,
        step=1,
        key="ask_ai_code_batch_size"
    )

    # On run
    if st.button("Run", key="ask_ai_code_run_btn"):
        # Prepare actual prompt by injecting JSON data
        prompt_filled = prompt_to_use.replace("{CODEBOOK_JSON_ARRAY}", codebook_json) \
                                     .replace("{JSON_DATA_BATCH}", df_data.to_json(orient="records")) \
                                     .replace("{TEXT_COLUMN_NAME}", selected_text_col) \
                                     .replace("{ID_COLUMN_NAME}", selected_id_col)
        # Run AI coding with spinner
        with st.spinner("AI coding in progress..."):
            ai_response = ai_services.generate_codes_with_codebook(
                ai_provider,
                prompt_filled,
                selected_id_col,
                ai_model,
                batch_size=st.session_state.ask_ai_code_batch_size
            )
        # Merge codes if response is a properly formatted list of dicts
        if isinstance(ai_response, list):
            try:
                for item in ai_response:
                    if not isinstance(item, dict):
                        raise ValueError(f"Invalid AI response item type: {type(item)}")
                    item_id = str(item.get(selected_id_col, ""))
                    codes_str = item.get("Codes", "")
                    mask = st.session_state.data_for_coding_tab3[selected_id_col].astype(str) == item_id
                    st.session_state.data_for_coding_tab3.loc[mask, "Codes"] = codes_str
                paths = st.session_state.currently_selected_view_filepaths_for_saving_tab3
                if len(paths) == 1:
                    df_save = st.session_state.data_for_coding_tab3.copy()
                    if "Source View" in df_save.columns:
                        df_save = df_save.drop(columns=["Source View"], errors="ignore")
                    data_management.save_coded_data_to_view(df_save, paths[0])
            except Exception as merge_err:
                logger.warning(f"Error merging AI response codes: {merge_err}")
        # Store response and trigger results dialog
        st.session_state["ai_code_response"] = ai_response
        st.session_state["show_ai_code_results"] = True
        st.rerun()


# --- AI Coding Results Dialog ---

@st.dialog("AI Coding Results", width="medium")
def ai_code_results_dialog():
    ai_response = st.session_state.get("ai_code_response")
    if isinstance(ai_response, list):
        st.success("AI coding applied successfully.")
        st.write("AI Response Log:")
        st.json(ai_response)
    else:
        st.error("AI failed to return valid code assignments.")
        st.write("Error Log:")
        st.write(ai_response)
    if st.button("Close AI Coding Results"):
        st.session_state["show_ai_code_results"] = False
        st.session_state.pop("ai_code_response", None)
        st.rerun()


# --- Initialize Session State Variables for this Page ---
if 'selected_project_views_info_tab3' not in st.session_state: st.session_state.selected_project_views_info_tab3 = {}
if 'data_for_coding_tab3' not in st.session_state: st.session_state.data_for_coding_tab3 = pd.DataFrame()
if 'data_for_row_wise_coding_actions' not in st.session_state: st.session_state.data_for_row_wise_coding_actions = pd.DataFrame()
if 'selected_rowwise_indices' not in st.session_state: st.session_state.selected_rowwise_indices = []
if 'selected_section2_indices' not in st.session_state: st.session_state.selected_section2_indices = []
if 'col_for_coding_tab3' not in st.session_state: st.session_state.col_for_coding_tab3 = "text"
if 'id_col_for_coding_tab3' not in st.session_state: st.session_state.id_col_for_coding_tab3 = "unique_app_id"
if 'search_term_coding_data_sec4' not in st.session_state: st.session_state.search_term_coding_data_sec4 = ""
if 'currently_selected_view_filepaths_for_saving_tab3' not in st.session_state: st.session_state.currently_selected_view_filepaths_for_saving_tab3 = []
if 'prompt_save_combined_as_new_view_tab3' not in st.session_state: st.session_state.prompt_save_combined_as_new_view_tab3 = False
if 'ai_batch_size_tab3' not in st.session_state: st.session_state.ai_batch_size_tab3 = 10
if 'ai_batch_prompt_template_p02' not in st.session_state:
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

if 'current_codebook_df' not in st.session_state:
    st.session_state.current_codebook_df = pd.DataFrame(columns=data_management.CODEBOOK_COLUMNS)
if 'edited_codebook_df' not in st.session_state: # This DF will now also include the "Select" column for the editor
    st.session_state.edited_codebook_df = pd.DataFrame(columns=["Select"] + data_management.CODEBOOK_COLUMNS)
if 'newly_added_code_names_codebook' not in st.session_state:
    st.session_state.newly_added_code_names_codebook = []
if 'ai_codebook_text_col' not in st.session_state: st.session_state.ai_codebook_text_col = None
if 'ai_codebook_id_col' not in st.session_state: st.session_state.ai_codebook_id_col = None
if 'ai_codebook_prompt_template' not in st.session_state:
    st.session_state.ai_codebook_prompt_template = """You are an experienced qualitative researcher analysing Reddit data.

GOAL
Create a concise, mutually-exclusive *codebook* that captures the main patterns across the entire dataset.

INSTRUCTIONS
1. Read every item’s "{TEXT_COLUMN_NAME}" value from the provided data.
2. Generate the smallest set of distinct codes that together cover the content.
   • Aim for ≈ min(√N, 30) codes (where N = number of posts, as provided in the context).
   • Each code label should be concise (ideally ≤ 3 words) and intuitive to humans.
3. For every code you generate, return an object with the following specific fields:
   • "Code":         The code label itself (string).
   • "Description":  A one-sentence definition of the code's meaning, explaining what kinds of posts or content segments it typically covers (string).
   • "Rationale":    A brief explanation of why this code is necessary or how it is distinct from other potential codes (string).
   • "Example_ids":  An array of up to 3 unique ID values (strings) from the "{ID_COLUMN_NAME}" field of the input data that best illustrate this code. These IDs must exist in the provided data.
4. Ensure that your generated codebook is comprehensive enough that every post in the dataset could be potentially exemplified by at least one code’s "Example_ids" if more examples were requested (though you only need to return up to 3 per code).
5. Do **not** invent meta-categories or broad themes at this stage; focus solely on generating a flat list of discrete codes for the codebook.

OUTPUT FORMAT
Return **only** a valid JSON array of these code objects. Do not include any introductory text, explanations, markdown formatting, or any characters or keys outside the main JSON array structure as specified.

EXAMPLE OF A SINGLE OUTPUT OBJECT:
{
  "Code": "Community Support",
  "Description": "Captures expressions of users offering or seeking help, advice, or encouragement from others within the community.",
  "Rationale": "Needed to identify instances of mutual aid and supportive interactions, distinct from general information seeking.",
  "Example_ids": ["post_id_123", "post_id_456", "post_id_789"]
}

DATA
The data you need to process is below:
```json
{JSON_DATA_BATCH}
```"""

if "inspect_dialog_selected_code_names" not in st.session_state: st.session_state.inspect_dialog_selected_code_names = []
if "inspect_dialog_example_ids_str_combined" not in st.session_state: st.session_state.inspect_dialog_example_ids_str_combined = ""
if "inspect_dialog_id_column" not in st.session_state: st.session_state.inspect_dialog_id_column = None


st.title("Coding & Codebook Development")

if not st.session_state.get('current_project_name') or not st.session_state.get('project_path'):
    st.warning("👈 Please create or open a project first from the '🏠 Project Setup' page.")
    st.stop()

if st.session_state.current_project_name and st.session_state.project_path:
    if 'codebook_loaded_for_project' not in st.session_state or \
       st.session_state.codebook_loaded_for_project != st.session_state.current_project_name:
        loaded_cb_df = data_management.load_codebook(st.session_state.project_path)
        # Add "Select" column to the loaded codebook for editor state management
        loaded_cb_df.insert(0, "Select", False)
        st.session_state.current_codebook_df = loaded_cb_df.copy() # Includes "Select"
        st.session_state.edited_codebook_df = loaded_cb_df.copy()   # Includes "Select"
        st.session_state.codebook_loaded_for_project = st.session_state.current_project_name
        st.session_state.newly_added_code_names_codebook = []


ai_provider_configured = st.session_state.project_config.get('ai_provider') and \
                         st.session_state.project_config.get(f"{st.session_state.project_config.get('ai_provider', '').lower()}_api", {}).get('api_key')

if not ai_provider_configured:
    st.warning("AI Provider and/or API Key not configured. Please set them on the '🏠 Project Setup' page to enable AI features.")

def robust_list_to_comma_string_p02(code_input):
    if isinstance(code_input, list): return ", ".join(str(c).strip() for c in code_input if str(c).strip())
    if pd.isna(code_input) or not str(code_input).strip(): return ""
    return str(code_input).strip()

def robust_comma_string_to_list_p02(code_str):
    if pd.isna(code_str) or not str(code_str).strip(): return []
    if isinstance(code_str, list): return [str(c).strip() for c in code_str if str(c).strip()]
    return [c.strip() for c in str(code_str).split(',') if c.strip()]

def load_and_combine_selected_views_for_coding_p02():
    # ... (this function remains the same as your last correct version)
    combined_df_list_views = []
    st.session_state.currently_selected_view_filepaths_for_saving_tab3 = []
    selected_view_infos = [ data for data in st.session_state.selected_project_views_info_tab3.values() if data.get("selected", False) ]
    if not selected_view_infos:
        st.session_state.data_for_coding_tab3 = pd.DataFrame()
        st.session_state.data_for_row_wise_coding_actions = pd.DataFrame()
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
        id_cols_present = [col for col in ['unique_app_id', 'id'] if col in concatenated_df.columns]
        if id_cols_present:
            dedup_col = 'unique_app_id' if 'unique_app_id' in id_cols_present else id_cols_present[0]
            concatenated_df.drop_duplicates(subset=[dedup_col], keep='first', inplace=True)
        else:
            concatenated_df.drop_duplicates(keep='first', inplace=True)
            logger.warning("No 'unique_app_id' or 'id' column for deduplication.")
        st.session_state.data_for_coding_tab3 = concatenated_df
        st.session_state.data_for_row_wise_coding_actions = concatenated_df.copy()
        st.session_state.data_for_row_wise_coding_actions['Codes'] = ""
        logger.info("Initialized 'data_for_row_wise_coding_actions' for Section 4.")
    else:
        st.session_state.data_for_coding_tab3 = pd.DataFrame()
        st.session_state.data_for_row_wise_coding_actions = pd.DataFrame()

st.subheader("1. Select Project View(s) for Coding")
available_views_meta_p02 = data_management.list_created_views_metadata()
if not available_views_meta_p02:
    st.info("No project views created yet. Go to '💾 Data Management' page to create a view.")
else:
    # Build DataFrame without custom 'Select' column
    views_list = []
    for view_meta_item in available_views_meta_p02:
        views_list.append({
            "View Name": view_meta_item["view_name"],
            "Created On": datetime.fromisoformat(view_meta_item.get("creation_timestamp", "")).strftime("%Y-%m-%d %H:%M") if view_meta_item.get("creation_timestamp") else "N/A",
            "Source Files Info": str(view_meta_item.get("source_files_info", "N/A"))
        })
    views_df = pd.DataFrame(views_list)

    # Use Streamlit's built-in row selections
    event = st.dataframe(
        views_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        height=min(300, (len(views_df) + 1) * 35 + 3)
    )

    # Update session state based on selected row indices
    selected_indices = event.selection.rows if hasattr(event, "selection") else []
    for idx, view_meta_item in enumerate(available_views_meta_p02):
        view_name = view_meta_item["view_name"]
        st.session_state.selected_project_views_info_tab3.setdefault(view_name, {})["selected"] = (idx in selected_indices)
        st.session_state.selected_project_views_info_tab3[view_name]["metadata"] = view_meta_item

    # Load and combine selected views for coding
    load_and_combine_selected_views_for_coding_p02()
st.divider()

st.subheader("2. Data to be Coded")
df_data_to_be_coded_sec2 = st.session_state.get('data_for_coding_tab3', pd.DataFrame())
if df_data_to_be_coded_sec2.empty:
    st.info("Select one or more views from the table above (Section 1) to load data.")
else:
    st.caption("This table shows the combined data selected from your views, including the **Codes** column where any applied codes will appear.")
    # Ensure 'Codes' column exists
    if 'Codes' not in df_data_to_be_coded_sec2.columns:
        df_data_to_be_coded_sec2['Codes'] = ""
    # Display selectable table for Section 2
    event_data2 = st.dataframe(
        df_data_to_be_coded_sec2,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        height=300,
        key="section2_table_display"
    )
    # Store selected rows
    selected_data_indices = event_data2.selection.rows if hasattr(event_data2, "selection") else []
    st.session_state.selected_section2_indices = selected_data_indices

    # Action buttons for Section 2: Update, Delete all codes, Ask AI to Code
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns([0.5, 0.15, 0.15, 0.2])
    # Placeholder column (text or spacing)
    with btn_col1:
        st.write("")  # Keeps layout alignment

    # Update codes button for single-row editing
    with btn_col2:
        update_disabled = len(st.session_state.selected_section2_indices) != 1
        if st.button("Update codes", key="update_codes_btn", disabled=update_disabled, use_container_width=True):
            update_codes_dialog()

    # Delete all codes button for multi-row clearing
    with btn_col3:
        delete_disabled = len(st.session_state.selected_section2_indices) < 1
        if st.button("Delete all codes", key="delete_all_codes_btn", disabled=delete_disabled, use_container_width=True):
            # Clear Codes for each selected row
            for row_idx in st.session_state.selected_section2_indices:
                st.session_state.data_for_coding_tab3.at[row_idx, "Codes"] = ""
            # Persist to underlying file if a single view is loaded
            selected_paths = st.session_state.get("currently_selected_view_filepaths_for_saving_tab3", [])
            if len(selected_paths) == 1:
                df_to_save = st.session_state.data_for_coding_tab3.copy()
                if "Source View" in df_to_save.columns:
                    df_to_save = df_to_save.drop(columns=["Source View"], errors="ignore")
                success = data_management.save_coded_data_to_view(df_to_save, selected_paths[0])
                if success:
                    ui_helpers.show_success_message(f"All codes deleted in '{os.path.basename(selected_paths[0])}'.")
                else:
                    ui_helpers.show_error_message(f"Failed to persist code deletions in '{os.path.basename(selected_paths[0])}'.")
            else:
                ui_helpers.show_info("All codes cleared in memory. Use 'Save All Row Codes' to persist.")
            st.rerun()


    # Ask AI to Code button
    with btn_col4:
        ask_ai_disabled = st.session_state.data_for_coding_tab3.empty or st.session_state.edited_codebook_df.drop(columns=["Select"], errors="ignore").empty
        if st.button("Ask AI to Code", key="ask_ai_code_btn", disabled=ask_ai_disabled, use_container_width=True):
            ask_ai_code_dialog()

# If AI coding has completed, show the results dialog (once)
if st.session_state.pop("show_ai_code_results", False):
    ai_code_results_dialog()

st.divider()



with st.expander("3. Codebook Development", expanded=True):
    st.subheader("3. Codebook Development")
    # >>> MOVE_TABLE_HERE >>>
    st.subheader("Current Codebook Draft")
    st.caption("Select one or more codebook entries below to inspect their examples.")
    # Inform about newly suggested AI codes
    if not st.session_state.edited_codebook_df.empty and st.session_state.newly_added_code_names_codebook:
        st.info(f"Newly suggested AI codes: {', '.join(st.session_state.newly_added_code_names_codebook)}. Please review.")
    # Prepare DataFrame for selection (drop internal columns)
    df_codebook = st.session_state.edited_codebook_df.copy()
    df_select = df_codebook.drop(columns=["Select", "Serial No."], errors="ignore").reset_index(drop=True)
    df_select.insert(0, "Serial No.", range(1, len(df_select) + 1))
    # Display table with built-in multi-row selection
    event_codebook = st.dataframe(
        df_select,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="multi-row",
        height=min(400, (len(df_select) + 1) * 35 + 3)
    )
    # Store selected indices in session state for inspection
    selected_indices = event_codebook.selection.rows if hasattr(event_codebook, "selection") else []
    st.session_state.selected_codebook_indices = selected_indices

    id_col_for_inspection_context = st.session_state.get('ai_codebook_id_col')
    if not id_col_for_inspection_context:
        if not df_data_to_be_coded_sec2.empty:
            if 'unique_app_id' in df_data_to_be_coded_sec2.columns:
                id_col_for_inspection_context = 'unique_app_id'
            elif 'id' in df_data_to_be_coded_sec2.columns:
                id_col_for_inspection_context = 'id'
    st.caption(f"For 'Inspect Examples', Example IDs will be matched against '`{id_col_for_inspection_context or 'N/A'}`' column of 'Data to be Coded' table.")

    # Section 3 action buttons in new order: View Examples, Edit Code, Delete Selected, Discard Changes, Save Changes, Apply Code
    btn_cols = st.columns(6)
    col_view, col_edit, col_delete, col_discard, col_save, col_apply = btn_cols

    # View Examples
    if col_view.button("View Examples", key="inspect_selected_codebook_examples_btn_main"):
        selected_indices = st.session_state.get("selected_codebook_indices", [])
        selected_rows_for_inspect = st.session_state.edited_codebook_df.iloc[selected_indices] if selected_indices else pd.DataFrame()
        if selected_rows_for_inspect.empty:
            ui_helpers.show_warning_message("No codes selected from the table for inspection.")
        else:
            all_example_ids_combined = []
            selected_code_names_for_dialog = selected_rows_for_inspect["Code Name"].tolist()
            for _, selected_row in selected_rows_for_inspect.iterrows():
                example_ids_for_this_code_str = str(selected_row.get('Example_ids', ''))
                if example_ids_for_this_code_str:
                    all_example_ids_combined.extend([eid.strip() for eid in example_ids_for_this_code_str.split(',') if eid.strip()])
            unique_example_ids_combined = sorted(list(set(all_example_ids_combined)))
            if not unique_example_ids_combined:
                ui_helpers.show_warning_message("Selected code(s) have no example IDs specified.")
            else:
                st.session_state.inspect_dialog_selected_code_names = selected_code_names_for_dialog
                st.session_state.inspect_dialog_example_ids_str_combined = ", ".join(unique_example_ids_combined)
                st.session_state.inspect_dialog_id_column = id_col_for_inspection_context
                inspect_codebook_examples_dialog()

    # Edit Code - enabled only when one row is selected
    edit_disabled = not (len(st.session_state.get("selected_codebook_indices", [])) == 1)
    if col_edit.button("Edit Code", key="edit_codebook_entry_btn", disabled=edit_disabled):
        st.session_state.edit_dialog_selected_index = st.session_state.selected_codebook_indices[0]
        edit_code_dialog()

    # Delete Selected
    if col_delete.button("Delete Selected", key="delete_selected_from_draft_btn"):
        selected_indices = st.session_state.get("selected_codebook_indices", [])
        selected_for_deletion = st.session_state.edited_codebook_df.iloc[selected_indices] if selected_indices else pd.DataFrame()
        if selected_for_deletion.empty:
            ui_helpers.show_warning_message("No codes selected from the table to delete.")
        else:
            indices_to_drop = selected_indices
            st.session_state.edited_codebook_df = st.session_state.edited_codebook_df.drop(indices_to_drop).reset_index(drop=True)
            # Persist deletion to saved codebook
            original_cb_df = st.session_state.current_codebook_df.copy()
            names_in_file = selected_for_deletion["Code Name"].tolist()
            if names_in_file:
                new_original_df = original_cb_df[~original_cb_df["Code Name"].isin(names_in_file)].copy()
                df_to_save = new_original_df.drop(columns=["Select"], errors="ignore").reset_index(drop=True)
                success = data_management.save_codebook(df_to_save, st.session_state.project_path)
                if success:
                    ui_helpers.show_success_message(f"Deleted {len(names_in_file)} entries from the saved codebook file.")
                    # Reset Select column to False for the updated codebook
                    new_original_df = new_original_df.drop(columns=["Select"], errors="ignore")
                    new_original_df.insert(0, "Select", False)
                    st.session_state.current_codebook_df = new_original_df
                else:
                    ui_helpers.show_error_message("Failed to delete entries from the saved codebook file.")
            ui_helpers.show_success_message(f"{len(indices_to_drop)} code(s) removed from draft. 'Save Changes' to make permanent.")
            st.rerun()

    # Discard Changes
    if col_discard.button("Discard Changes", key="discard_codebook_changes_main"):
        loaded_cb_df_discard = data_management.load_codebook(st.session_state.project_path)
        loaded_cb_df_discard.insert(0, "Select", False)
        st.session_state.current_codebook_df = loaded_cb_df_discard.copy()
        st.session_state.edited_codebook_df = loaded_cb_df_discard.copy()
        st.session_state.newly_added_code_names_codebook = []
        st.info("Codebook changes discarded and reloaded from saved version.")
        st.rerun()

    # Save Changes
    if col_save.button("Save Changes", key="accept_codebook_changes_main"):
        df_to_save_codebook = st.session_state.edited_codebook_df.drop(columns=["Select"], errors="ignore").reset_index(drop=True)
        if data_management.save_codebook(df_to_save_codebook, st.session_state.project_path):
            loaded_cb_for_edit = df_to_save_codebook.copy()
            loaded_cb_for_edit.insert(0, "Select", False)
            st.session_state.current_codebook_df = loaded_cb_for_edit.copy()
            st.session_state.edited_codebook_df = loaded_cb_for_edit.copy()
            st.session_state.newly_added_code_names_codebook = []
            ui_helpers.show_success_message("Codebook saved successfully.")
            st.rerun()
        else:
            ui_helpers.show_error_message("Failed to save codebook.")

    # Apply Code to selected data rows
    if col_apply.button("Apply Code", key="add_code_section3_btn",
                        disabled=(not st.session_state.selected_section2_indices or not st.session_state.selected_codebook_indices)):
        df_to_update = st.session_state.data_for_coding_tab3.copy()
        code_names = st.session_state.edited_codebook_df.loc[
            st.session_state.selected_codebook_indices, 'Code Name'
        ].tolist()
        for row_idx in st.session_state.selected_section2_indices:
            existing_codes = robust_comma_string_to_list_p02(df_to_update.at[row_idx, 'Codes'])
            for code in code_names:
                if code not in existing_codes:
                    existing_codes.append(code)
            df_to_update.at[row_idx, 'Codes'] = robust_list_to_comma_string_p02(existing_codes)
        st.session_state.data_for_coding_tab3 = df_to_update
        selected_paths = st.session_state.get('currently_selected_view_filepaths_for_saving_tab3', [])
        if len(selected_paths) == 1:
            df_to_save = df_to_update.copy()
            if 'Source View' in df_to_save.columns:
                df_to_save = df_to_save.drop(columns=['Source View'], errors='ignore')
            success = data_management.save_coded_data_to_view(df_to_save, selected_paths[0])
            if success:
                ui_helpers.show_success_message(f"Codes saved to '{os.path.basename(selected_paths[0])}'.")
            else:
                ui_helpers.show_error_message(f"Failed to save codes to '{os.path.basename(selected_paths[0])}'.")
        else:
            ui_helpers.show_info("Codes updated in-memory. Use 'Save All Row Codes' to persist to files.")
        st.rerun()

    # ... (AI and Manual Codebook Entry Tabs - largely unchanged, except for how edited_codebook_df is populated)
    # The key change will be in how `edited_codebook_df_with_select` is managed and used.
    if 'current_project_name' in st.session_state and st.session_state.project_path:
        if 'codebook_loaded_for_project' not in st.session_state or st.session_state.codebook_loaded_for_project != st.session_state.current_project_name:
            # This logic is fine, ensures "Select" is added when loading/reloading
            loaded_cb_df = data_management.load_codebook(st.session_state.project_path)
            if "Select" not in loaded_cb_df.columns: loaded_cb_df.insert(0, "Select", False)
            st.session_state.current_codebook_df = loaded_cb_df.copy()
            st.session_state.edited_codebook_df = loaded_cb_df.copy()
            st.session_state.codebook_loaded_for_project = st.session_state.current_project_name
            st.session_state.newly_added_code_names_codebook = []

    col_mode, col_spacer = st.columns([1, 2])
    codebook_view_option = col_mode.selectbox(
        "Choose Codebook Mode:",
        ["AI-Generated Codebook", "Manual Codebook Entry"],
        index=0,
        key="codebook_mode_select"
    )
    if codebook_view_option == "AI-Generated Codebook":
        st.markdown("Use AI to generate an initial codebook based on the **entire 'Data to be Coded' table** (Section 2).")
        if not ai_provider_configured: st.info("AI features disabled.")
        elif df_data_to_be_coded_sec2.empty: st.info("Load data into 'Data to be Coded' (Section 2) first.")
        else:
            ai_codebook_cols_cfg = st.columns(2)
            text_col_options_codebook = [col for col in df_data_to_be_coded_sec2.columns if df_data_to_be_coded_sec2[col].dtype == 'object' and col not in ['Source View', 'unique_app_id', 'id']]
            id_col_options_codebook = [col for col in ['unique_app_id', 'id'] if col in df_data_to_be_coded_sec2.columns]
            if not text_col_options_codebook: st.warning("No suitable text columns found for AI codebook generation.")
            if not id_col_options_codebook: st.warning("No suitable ID column found for AI codebook generation.")
            if text_col_options_codebook and id_col_options_codebook:
                current_ai_cb_text_col = st.session_state.get('ai_codebook_text_col')
                current_ai_cb_id_col = st.session_state.get('ai_codebook_id_col')
                default_text_idx_cb = 0
                if current_ai_cb_text_col and current_ai_cb_text_col in text_col_options_codebook: default_text_idx_cb = text_col_options_codebook.index(current_ai_cb_text_col)
                elif 'text' in text_col_options_codebook: default_text_idx_cb = text_col_options_codebook.index('text')
                default_id_idx_cb = 0
                if current_ai_cb_id_col and current_ai_cb_id_col in id_col_options_codebook: default_id_idx_cb = id_col_options_codebook.index(current_ai_cb_id_col)
                elif 'unique_app_id' in id_col_options_codebook: default_id_idx_cb = id_col_options_codebook.index('unique_app_id')
                st.session_state.ai_codebook_text_col = ai_codebook_cols_cfg[0].selectbox("Select Text Column for AI Codebook Input:", text_col_options_codebook, index=default_text_idx_cb, key="ai_codebook_text_col_select_widget_ai_tab")
                st.session_state.ai_codebook_id_col = ai_codebook_cols_cfg[1].selectbox("Select ID Column for AI Codebook Input (used for Example_ids):", id_col_options_codebook, index=default_id_idx_cb, key="ai_codebook_id_col_select_widget_ai_tab")
                ai_provider_codebook = st.session_state.project_config.get('ai_provider', "OpenAI")
                ai_model_codebook = None
                if ai_provider_codebook == "OpenAI":
                    ai_model_codebook = st.selectbox("OpenAI Model (Codebook):", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], key="openai_model_codebook_p02_ai_tab_model")
                elif ai_provider_codebook == "Gemini":
                    ai_model_codebook = st.selectbox(
                        "Gemini Model (Codebook):",
                        ["gemini-2.0-flash", "gemini-1.5-flash"],
                        index=0,
                        key="gemini_model_codebook_p02_ai_tab_model"
                    )
                st.text_area("AI Codebook Prompt Template:", value=st.session_state.ai_codebook_prompt_template, height=300, key="ai_codebook_prompt_input_ai_tab_prompt", on_change=lambda: setattr(st.session_state, 'ai_codebook_prompt_template', st.session_state.ai_codebook_prompt_input_ai_tab_prompt))
                N_codebook = len(df_data_to_be_coded_sec2)
                target_codes_info = min(round(math.sqrt(N_codebook)), 30) if N_codebook > 0 else 0
                st.caption(f"Prompt guides AI for ≈ {target_codes_info} codes for {N_codebook} items. AI uses '{st.session_state.ai_codebook_id_col}' for 'Example_ids'.")
                # Batch size for AI codebook generation
                st.number_input(
                    "Batch Size for AI Codebook Generation:",
                    min_value=1,
                    value=st.session_state.ai_batch_size_tab3,
                    step=1,
                    help="Maximum number of items sent to the AI in each batch.",
                    key="ai_batch_size_tab3"
                )
                if st.button("Generate/Update Codebook with AI", key="generate_ai_codebook_btn_ai_tab_action"):
                    if not st.session_state.ai_codebook_text_col or not st.session_state.ai_codebook_id_col or not ai_model_codebook: ui_helpers.show_error_message("Text column, ID column, and AI model must be selected.")
                    else:
                        with st.spinner("AI is generating codebook..."):
                            ai_response = ai_services.generate_codebook_with_ai(
                                df_data_to_be_coded_sec2,
                                ai_provider_codebook,
                                st.session_state.ai_codebook_prompt_template,
                                st.session_state.ai_codebook_text_col,
                                st.session_state.ai_codebook_id_col,
                                ai_model_codebook,
                                st.session_state.ai_batch_size_tab3
                            )
                        if isinstance(ai_response, list) and all(isinstance(item, dict) for item in ai_response):
                            new_cb_entries_df = pd.DataFrame(ai_response)
                            for col_cb_ai in data_management.CODEBOOK_COLUMNS:
                                if col_cb_ai not in new_cb_entries_df.columns: new_cb_entries_df[col_cb_ai] = ""
                            new_cb_entries_df = new_cb_entries_df[data_management.CODEBOOK_COLUMNS] # Ensure order and existence
                            if 'Example_ids' in new_cb_entries_df.columns: new_cb_entries_df['Example_ids'] = new_cb_entries_df['Example_ids'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else (str(x) if pd.notna(x) else ''))
                            
                            # Add "Select" column for newly generated AI codes
                            new_cb_entries_df.insert(0, "Select", False)

                            # Append to existing edited_codebook_df (which already has "Select")
                            # Ensure no duplicate "Select" columns if edited_codebook_df somehow got it removed
                            if "Select" not in st.session_state.edited_codebook_df.columns and not st.session_state.edited_codebook_df.empty:
                                 st.session_state.edited_codebook_df.insert(0, "Select", False)
                            elif st.session_state.edited_codebook_df.empty: # If starting fresh
                                 st.session_state.edited_codebook_df = pd.DataFrame(columns=["Select"] + data_management.CODEBOOK_COLUMNS)


                            st.session_state.edited_codebook_df = pd.concat([st.session_state.edited_codebook_df, new_cb_entries_df], ignore_index=True).drop_duplicates(subset=['Code Name'], keep='last')
                            st.session_state.newly_added_code_names_codebook = new_cb_entries_df["Code Name"].tolist()
                            ui_helpers.show_success_message(f"AI suggested {len(new_cb_entries_df)} entries.")
                            st.rerun()
                        elif isinstance(ai_response, dict) and "error" in ai_response: ui_helpers.show_error_message(f"AI Error: {ai_response['error']}")
                        else: ui_helpers.show_error_message("AI failed to return valid data.")
            else: st.info("Ensure 'Data to be Coded' (Section 2) has text and ID columns.")

    elif codebook_view_option == "Manual Codebook Entry":
        st.markdown("Manually add or edit entries in the codebook.")
        with st.form("manual_codebook_entry_form_manual_tab"):
            st.subheader("Add New Codebook Entry")
            new_code_name = st.text_input("Code Name*", key="cb_manual_code_name_manual_tab_input")
            new_description = st.text_area("Description", key="cb_manual_desc_manual_tab_input")
            new_rationale = st.text_area("Rationale", key="cb_manual_rationale_manual_tab_input")
            new_example_ids = st.text_input(f"Example IDs (comma-separated, from an ID column in 'Data to be Coded')", key="cb_manual_ex_ids_manual_tab_input")
            submitted_manual_codebook = st.form_submit_button("Add Entry to Codebook Draft")
            if submitted_manual_codebook:
                if not new_code_name.strip(): ui_helpers.show_error_message("Code Name is required.")
                # Check against Code Name in edited_codebook_df (which includes "Select")
                elif new_code_name.strip() in st.session_state.edited_codebook_df["Code Name"].values:
                    ui_helpers.show_error_message(f"Code Name '{new_code_name.strip()}' already exists.")
                else:
                    new_entry_manual = pd.DataFrame([{
                        "Select": False, # Add "Select" column for new manual entries
                        "Code Name": new_code_name.strip(),
                        "Description": new_description.strip(),
                        "Rationale": new_rationale.strip(),
                        "Example_ids": new_example_ids.strip()
                    }])
                    # Ensure edited_codebook_df has "Select" if it was empty before
                    if st.session_state.edited_codebook_df.empty:
                        st.session_state.edited_codebook_df = pd.DataFrame(columns=["Select"] + data_management.CODEBOOK_COLUMNS)

                    st.session_state.edited_codebook_df = pd.concat([st.session_state.edited_codebook_df, new_entry_manual], ignore_index=True)
                    ui_helpers.show_success_message(f"Entry '{new_code_name.strip()}' added to draft.")
                    st.rerun()

st.divider()