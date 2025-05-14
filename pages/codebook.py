# pages/codebook.py
# Persist global theme and layout on this page
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from modules import data_manager, ui_helpers, utils, ai_services
import math # For sqrt in codebook generation prompt info
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid

logger = utils.setup_logger("p02_ai_coding")
# Cosine-similarity threshold for duplicate code detection
DUP_SIM_THRESHOLD = 0.90
# Try to load the model with ONNXRuntime + dynamic‑INT8 quantization (CPU‑optimized).
# Requires: pip install "sentence-transformers[onnx]"
try:
    _ST_MODEL = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        trust_remote_code=True,          # needed for some ONNX configs
        quantize="dynamic"               # <─ triggers 8‑bit quant + ONNX if deps are present
    )
    logger.info("Loaded MiniLM with ONNXRuntime (INT8 dynamic quant).")
except Exception as onnx_err:
    logger.warning(f"ONNX/quantized load failed ({onnx_err}). Falling back to default PyTorch model.")
    _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

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
        # Fallback suffix matching if no direct matches (handles timestamp prefix differences)
        if filtered_df.empty:
            suffixes = [eid.split('_', 1)[1] if '_' in eid else eid for eid in example_ids_list]
            mask = data_to_filter[actual_id_col].astype(str).apply(
                lambda x: any(x.endswith(suf) for suf in suffixes)
            )
            filtered_df = data_to_filter[mask]
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
        # Update code_ids column to match selected codes
        code_ids_list = []
        for code in selected_codes:
            meta = st.session_state.edited_codebook_df[
                st.session_state.edited_codebook_df["Code Name"] == code
            ]
            if not meta.empty:
                code_ids_list.append(str(meta.iloc[0]["code_id"]))
        st.session_state.data_for_coding_tab3.at[idx, "code_ids"] = ", ".join(code_ids_list)
        # Persist to underlying file if exactly one view is loaded
        selected_paths = st.session_state.get("currently_selected_view_filepaths_for_saving_tab3", [])
        if len(selected_paths) == 1:
            df_to_save = st.session_state.data_for_coding_tab3.copy()
            if "Source View" in df_to_save.columns:
                df_to_save = df_to_save.drop(columns=["Source View"], errors="ignore")
            success = data_manager.save_coded_data_to_view(df_to_save, selected_paths[0])
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
    # Ensure code_ids column exists for tracking code_id values
    if 'code_ids' not in st.session_state.data_for_coding_tab3.columns:
        st.session_state.data_for_coding_tab3['code_ids'] = ""
    # Prepare inputs
    df_data = st.session_state.data_for_coding_tab3.copy()
    # Include code_id in the JSON so AI can reference it
    codebook_df = st.session_state.edited_codebook_df.drop(columns=["Select"], errors="ignore").reset_index(drop=True)
    codebook_json = codebook_df.to_json(orient="records")
    # Column selectors
    text_columns = [col for col in df_data.columns if df_data[col].dtype == "object" and col not in ["Source View", "Codes"]]
    id_columns = [col for col in ["unique_app_id", "id"] if col in df_data.columns]
    default_text_index = text_columns.index("text") if "text" in text_columns else 0
    selected_text_col = st.selectbox(
        "Select Text Column:",
        text_columns,
        index=default_text_index,
        key="ask_ai_code_text_col"
    )
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
3. Output an object with exactly three keys:
   • "{ID_COLUMN_NAME}" — the original identifier (string)
   • "Codes" — a single comma-separated string of the applicable Code labels,
               or an empty string if none fit.
   • "code_ids" — a single comma-separated string of the corresponding code_id values
                  (from the codebook) for the Codes, or an empty string if no codes applied.

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
                    # Populate code_ids: use AI-provided or derive from codebook
                    code_ids_str = item.get("code_ids", "")
                    if (not code_ids_str) and codes_str:
                        # Derive code_ids by looking up each code in the codebook
                        derived_ids = []
                        for cd in codes_str.split(","):
                            cd_clean = cd.strip()
                            meta = st.session_state.edited_codebook_df[
                                st.session_state.edited_codebook_df["Code Name"] == cd_clean
                            ]
                            if not meta.empty:
                                derived_ids.append(str(meta.iloc[0]["code_id"]))
                        code_ids_str = ", ".join(derived_ids)
                    st.session_state.data_for_coding_tab3.loc[mask, "code_ids"] = code_ids_str
                paths = st.session_state.currently_selected_view_filepaths_for_saving_tab3
                if len(paths) == 1:
                    df_save = st.session_state.data_for_coding_tab3.copy()
                    if "Source View" in df_save.columns:
                        df_save = df_save.drop(columns=["Source View"], errors="ignore")
                    data_manager.save_coded_data_to_view(df_save, paths[0])
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


# --- Dialog: AI Codebook Generation Modal ---
@st.dialog("Generate AI Codebook", width="large")
def ai_codebook_generation_modal():
    # Prepare inputs
    df_data = st.session_state.get('data_for_coding_tab3', pd.DataFrame()).copy()
    if df_data.empty:
        st.error("Load data into 'Data to be Coded' (Section 2) first.")
        if st.button("Close"):
            st.rerun()
        return
    # Column selectors
    text_cols = [col for col in df_data.columns if df_data[col].dtype == "object" and col not in ["Source View", "Codes"]]
    id_cols = [col for col in ["unique_app_id", "id"] if col in df_data.columns]
    selected_text = st.selectbox("Select Text Column for AI Codebook Input:", text_cols, index=0, key="ai_cb_text_col")
    selected_id = st.selectbox("Select ID Column for AI Codebook Input (for Example_ids):", id_cols, index=0, key="ai_cb_id_col")
    # AI provider and model
    provider = st.session_state.project_config.get('ai_provider', 'OpenAI')
    if provider == 'OpenAI':
        model = st.selectbox("OpenAI Model (Codebook):", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], key="ai_model_cb_openai")
    else:
        model = st.selectbox("Gemini Model (Codebook):", ["gemini-2.0-flash", "gemini-1.5-flash"], key="ai_model_cb_gemini")
    # ---- Build FULL prompt that includes current codebook JSON so user sees everything ----
    existing_codebook_json = st.session_state.edited_codebook_df.drop(columns=["Select", "code_id"], errors="ignore").to_json(orient="records")
    merge_context = (
        "CURRENT CODEBOOK (Draft)\n"
        "The JSON array below is the existing draft codebook. "
        "• If you propose a code that already exists (same or very similar meaning), "
        "merge with it by updating its Description/Rationale/Example_ids instead of adding a duplicate. "
        "• Only create new codes for genuinely new concepts.\n\n"
        f"{existing_codebook_json}\n\n"
    )
    default_full_prompt = merge_context + st.session_state.ai_codebook_prompt_template
    prompt = st.text_area(
        "Full AI Codebook Prompt (editable):",
        value=default_full_prompt,
        height=400,
        key="ai_cb_full_prompt"
    )
    # sample/batching controls
    send_samples = st.checkbox("Send Samples", key="ai_cb_send_samples")
    if send_samples:
        sample_percentage = st.slider(
            "Percentage of samples to send (%)", min_value=5, max_value=10, value=5, key="ai_cb_sample_percentage"
        )
        sample_count = math.ceil(len(df_data) * sample_percentage / 100)
        st.write(f"Sending {sample_count} samples out of {len(df_data)}.")
    else:
        sample_count = None

    batch_query = st.checkbox("Batch Query", key="ai_cb_batch_query")
    if batch_query:
        batch_size = st.number_input(
            "Batch Size for AI Codebook Generation:",
            min_value=1,
            value=st.session_state.ai_batch_size_tab3,
            step=1,
            key="ai_cb_batch_size"
        )
    else:
        batch_size = len(df_data) if sample_count is None else sample_count
    if st.button("Generate/Update Codebook with AI", key="ai_generate_codebook_btn"):
        # Gather current code names so the AI can avoid duplicates in subsequent batches
        existing_code_names = st.session_state.edited_codebook_df["Code Name"].dropna().unique().tolist()
        df_for_ai = (
            df_data.sample(n=sample_count, random_state=42)
            if send_samples and sample_count
            else df_data
        )
        with st.spinner("AI is generating codebook..."):
            response = ai_services.generate_codebook_with_ai(
                df_for_ai,
                provider,
                prompt,
                selected_text,
                selected_id,
                model,
                batch_size,
                existing_code_names
            )
        # Handle response (same as original code)
        if isinstance(response, list) and all(isinstance(item, dict) for item in response):
            new_df = pd.DataFrame(response)
            # Rename "Code" key from AI response to "Code Name"
            if 'Code' in new_df.columns:
                new_df = new_df.rename(columns={'Code': 'Code Name'})
            # Assign a unique code_id to each new code entry
            new_df['code_id'] = [str(uuid.uuid4()) for _ in range(len(new_df))]
            for col in data_manager.CODEBOOK_COLUMNS:
                if col not in new_df.columns: new_df[col] = ""
            new_df = new_df[data_manager.CODEBOOK_COLUMNS]
            new_df['Example_ids'] = new_df['Example_ids'].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else (str(x) if pd.notna(x) else ''))
            new_df.insert(0, 'Select', False)
            # Append to draft
            if 'Select' not in st.session_state.edited_codebook_df.columns:
                st.session_state.edited_codebook_df.insert(0, 'Select', False)
            st.session_state.edited_codebook_df = pd.concat([st.session_state.edited_codebook_df, new_df], ignore_index=True).drop_duplicates(subset=['Code Name'], keep='last')
            st.session_state.newly_added_code_names_codebook = new_df['Code Name'].tolist()
            ui_helpers.show_success_message(f"AI suggested {len(new_df)} entries.")
            st.rerun()
        elif isinstance(response, dict) and 'error' in response:
            ui_helpers.show_error_message(f"AI Error: {response['error']}")
        else:
            ui_helpers.show_error_message("AI failed to return valid data.")
    if st.button("Close AI Codebook Generator"):
        st.session_state.show_ai_codebook_review_dialog = True
        st.rerun()


# --- Dialog: Manual Codebook Entry Modal ---
@st.dialog("Add Manual Codebook Entry", width="large")
def manual_codebook_entry_modal():
    st.subheader("Add New Codebook Entry")
    with st.form("manual_codebook_entry_form_modal"):
        name = st.text_input("Code Name*", key="modal_manual_code_name")
        desc = st.text_area("Description", key="modal_manual_desc")
        rationale = st.text_area("Rationale", key="modal_manual_rationale")
        ex_ids = st.text_input("Example IDs (comma-separated)", key="modal_manual_ex_ids")
        submitted = st.form_submit_button("Add Entry")
        if submitted:
            if not name.strip():
                ui_helpers.show_error_message("Code Name is required.")
            elif name.strip() in st.session_state.edited_codebook_df['Code Name'].values:
                ui_helpers.show_error_message(f"Code Name '{name.strip()}' already exists.")
            else:
                new_entry = pd.DataFrame([{
                    'Select': False,
                    'code_id': str(uuid.uuid4()),
                    'Code Name': name.strip(),
                    'Description': desc.strip(),
                    'Rationale': rationale.strip(),
                    'Example_ids': ex_ids.strip()
                }])
                if st.session_state.edited_codebook_df.empty:
                    st.session_state.edited_codebook_df = pd.DataFrame(columns=['Select'] + data_manager.CODEBOOK_COLUMNS)
                st.session_state.edited_codebook_df = pd.concat([st.session_state.edited_codebook_df, new_entry], ignore_index=True)
                ui_helpers.show_success_message(f"Entry '{name.strip()}' added to draft.")
                st.rerun()
    if st.button("Close Manual Entry"):
        st.rerun()


# --- Dialog: AI Codebook Review Suggestions Modal ---
@st.dialog("Review AI Codebook Suggestions", width="medium")
def ai_codebook_review_dialog():
    new_codes = st.session_state.get("newly_added_code_names_codebook", [])
    if new_codes:
        msg = f"Newly suggested AI codes: {', '.join(new_codes)}. Please review."
    else:
        msg = "No new AI codes were suggested."
    st.info(msg)
    if st.button("Close"):
        st.session_state.newly_added_code_names_codebook = []
        st.rerun()

# --- Dialog: Find & Merge Duplicate Codes -------------------------------
@st.dialog("Find & Merge Duplicate Codes", width="large")
def duplicate_codes_dialog():
    """
    Detect semantically similar code labels using OpenAI embeddings +
    cosine similarity, show them in clusters, and let the user merge them.
    """
    code_names = st.session_state.edited_codebook_df["Code Name"].dropna().tolist()
    clusters = _find_duplicate_clusters(
        st.session_state.edited_codebook_df.drop(columns=["Select"], errors="ignore"),
        st.session_state.get("data_for_coding_tab3", pd.DataFrame()),
        id_col_for_inspection_context or "unique_app_id",
        st.session_state.get("ai_codebook_text_col") or st.session_state.get("col_for_coding_tab3") or "text"
    )

    if not clusters:
        st.success("No semantic duplicates detected 🎉")
        if st.button("Close"):
            st.rerun()
        return

    st.subheader("Possible Duplicate Code Clusters")
    st.caption("Pick ONE label to keep per cluster. Others will be merged into it.")
    replacement_map = {}  # dup_name -> primary_name

    for idx, cluster in enumerate(clusters, 1):
        st.write(f"**Cluster {idx}**")
        keep_all_token = "⏸️ Keep all (no merge)"
        keep = st.radio(
            f"Choose what to keep for Cluster {idx}",
            options=[keep_all_token] + [c["name"] for c in cluster],
            horizontal=True,
            key=f"dup_radio_{idx}"
        )

        tabs = st.tabs([c["name"] for c in cluster])
        for tab, c in zip(tabs, cluster):
            with tab:
                st.caption(f"Similarity to seed: **{c['sim']:.2f}**")
                st.markdown(f"**Description:** {c['description']}")
                st.markdown(f"**Rationale:** {c['rationale']}")
                st.markdown(f"**Example IDs:** {c['examples']}")
                if c["example_text"]:
                    with st.container(height=160):
                        st.write(c["example_text"])
        if keep != keep_all_token:
            for c in cluster:
                if c["name"] != keep:
                    replacement_map[c["name"]] = keep
        st.divider()

    if st.button("Merge Selected Duplicates"):
        # 1) Update codebook in memory
        df_cb = st.session_state.edited_codebook_df.copy()
        df_cb["Code Name"] = df_cb["Code Name"].apply(lambda x: replacement_map.get(x, x))
        df_cb = df_cb.drop_duplicates(subset=["Code Name"], keep="first").reset_index(drop=True)
        st.session_state.edited_codebook_df = df_cb

        # 2) Update any codes already assigned to rows
        if not st.session_state.data_for_coding_tab3.empty:
            df_data = st.session_state.data_for_coding_tab3.copy()

            def _replace_codes_str(code_str):
                codes = robust_comma_string_to_list_p02(code_str)
                codes = [replacement_map.get(c, c) for c in codes]
                return robust_list_to_comma_string_p02(sorted(set(codes)))

            df_data["Codes"] = df_data["Codes"].apply(_replace_codes_str)
            st.session_state.data_for_coding_tab3 = df_data

        ui_helpers.show_success_message(f"Merged {len(replacement_map)} duplicate label(s).")
        st.rerun()

    if st.button("Close"):
        st.rerun()
# ------------------------------------------------------------------------


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
    st.session_state.current_codebook_df = pd.DataFrame(columns=data_manager.CODEBOOK_COLUMNS)
if 'edited_codebook_df' not in st.session_state: # This DF will now also include the "Select" column for the editor
    st.session_state.edited_codebook_df = pd.DataFrame(columns=["Select"] + data_manager.CODEBOOK_COLUMNS)
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


st.title("Codebook")

if not st.session_state.get('current_project_name') or not st.session_state.get('project_path'):
    st.warning("👈 Please create or open a project first from the '🏠 Project Setup' page.")
    st.stop()

if st.session_state.current_project_name and st.session_state.project_path:
    if 'codebook_loaded_for_project' not in st.session_state or \
       st.session_state.codebook_loaded_for_project != st.session_state.current_project_name:
        loaded_cb_df = data_manager.load_codebook(st.session_state.project_path)
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

# ---------- Duplicate-detection helpers ----------
def _compute_embeddings(text_list):
    """
    Return a list of 384‑dimensional embeddings for each string using the
    local MiniLM model. Guaranteed no external API cost.
    """
    try:
        return _ST_MODEL.encode(text_list, convert_to_numpy=True).tolist()
    except Exception as err:
        logger.error(f"Local embedding generation failed: {err}")
        dim = 384
        return [np.zeros(dim).tolist() for _ in text_list]

def _find_duplicate_clusters(
    codebook_df: pd.DataFrame,
    data_df: pd.DataFrame,
    id_column: str,
    text_column: str
):
    """
    Cluster potentially duplicate codes **based on Code Name, Description, and Rationale only**. Example comment text is *not* included in the similarity computation but is still returned for user review.
    Each cluster is a list of dicts:
      {
        "name":        Code Name,
        "sim":         cosine similarity to cluster seed,
        "description": Description,
        "rationale":   Rationale,
        "examples":    Example_ids (string),
        "example_text": concatenated comment text (≤500 chars)
      }
    """
    if codebook_df.empty or data_df.empty:
        return []

    # Grab example comment text
    def _example_text(ids_cell):
        ids = [s.strip() for s in str(ids_cell).split(",") if s.strip()]
        subset = data_df[data_df[id_column].astype(str).isin(ids)]
        if subset.empty and ids:
            suf = [sid.split("_", 1)[1] if "_" in sid else sid for sid in ids]
            subset = data_df[data_df[id_column].astype(str).apply(
                lambda x: any(x.endswith(s) for s in suf)
            )]
        return " ".join(subset[text_column].astype(str).tolist())

    composite = (
        codebook_df["Code Name"].fillna("") + " " +
        codebook_df["Description"].fillna("") + " " +
        codebook_df["Rationale"].fillna("")
    ).str.strip().tolist()

    vecs = np.array(_compute_embeddings(composite))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-8)
    sims = vecs @ vecs.T
    names = codebook_df["Code Name"].tolist()

    visited, clusters = set(), []
    for i, name in enumerate(names):
        if i in visited:
            continue
        row_sim = sims[i]
        dup_idx = [j for j in range(i + 1, len(names))
                   if j not in visited and row_sim[j] >= DUP_SIM_THRESHOLD]
        if dup_idx:
            cluster_idx = [i] + dup_idx
            visited.update(cluster_idx)
            cluster = []
            for j in cluster_idx:
                cluster.append({
                    "name": names[j],
                    "sim": round(float(row_sim[j]), 3),
                    "description": str(codebook_df.at[j, "Description"]),
                    "rationale": str(codebook_df.at[j, "Rationale"]),
                    "examples": str(codebook_df.at[j, "Example_ids"]),
                    "example_text": _example_text(codebook_df.at[j, "Example_ids"])
                })
            clusters.append(cluster)
    return clusters
# -------------------------------------------------

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
            df_single_view = data_manager.load_data_from_specific_file(csv_path)
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
available_views_meta_p02 = data_manager.list_created_views_metadata()
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
    # Ensure code_ids column is present in the displayed table
    if 'code_ids' not in df_data_to_be_coded_sec2.columns:
        df_data_to_be_coded_sec2['code_ids'] = ""
    # Prepare display DataFrame with serial numbers
    df_display_sec2 = df_data_to_be_coded_sec2.copy().reset_index(drop=True)
    df_display_sec2.insert(0, "Serial No.", range(1, len(df_display_sec2) + 1))
    event_data2 = st.dataframe(
        df_display_sec2,
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
                st.session_state.data_for_coding_tab3.at[row_idx, "code_ids"] = ""
            # Persist to underlying file if a single view is loaded
            selected_paths = st.session_state.get("currently_selected_view_filepaths_for_saving_tab3", [])
            if len(selected_paths) == 1:
                df_to_save = st.session_state.data_for_coding_tab3.copy()
                if "Source View" in df_to_save.columns:
                    df_to_save = df_to_save.drop(columns=["Source View"], errors="ignore")
                success = data_manager.save_coded_data_to_view(df_to_save, selected_paths[0])
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



st.subheader("3. Codebook Development")
# >>> MOVE_TABLE_HERE >>>
st.subheader("Current Codebook Draft")
st.caption("Select one or more codebook entries below to inspect their examples.")
# Buttons for codebook entry
col_ai, col_manual = st.columns(2)
if col_ai.button("Ask AI to Generate Codebook", use_container_width=True):
    ai_codebook_generation_modal()
if col_manual.button("Manual Codebook Entry", use_container_width=True):
    manual_codebook_entry_modal()
# Prepare DataFrame for selection (drop internal columns)
df_codebook = st.session_state.edited_codebook_df.copy()
df_select = df_codebook.drop(columns=["Select", "Serial No."], errors="ignore").reset_index(drop=True)
df_select.insert(0, "Serial No.", range(1, len(df_select) + 1))
# Highlight draft codes (entries not yet saved) with a light blue background
saved_names = set(st.session_state.current_codebook_df["Code Name"].tolist())
styled_select = df_select.style.apply(
    lambda row: ["background-color: #e6f2ff" if row["Code Name"] not in saved_names else "" for _ in row],
    axis=1
)
# Display table with built-in multi-row selection
event_codebook = st.dataframe(
    styled_select,
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

# Section 3 action buttons in new order: View Examples, Edit Code, Find Duplicates, Delete Selected, Discard Changes, Save Changes, Apply Code
# Add an extra column for the new "Find Duplicates" action
btn_cols = st.columns(7)
col_view, col_edit, col_dupfind, col_delete, col_discard, col_save, col_apply = btn_cols

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

# Find Duplicates
if col_dupfind.button("Find Duplicates", key="find_duplicates_btn"):
    duplicate_codes_dialog()

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
            success = data_manager.save_codebook(df_to_save, st.session_state.project_path)
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
    loaded_cb_df_discard = data_manager.load_codebook(st.session_state.project_path)
    loaded_cb_df_discard.insert(0, "Select", False)
    st.session_state.current_codebook_df = loaded_cb_df_discard.copy()
    st.session_state.edited_codebook_df = loaded_cb_df_discard.copy()
    st.session_state.newly_added_code_names_codebook = []
    st.info("Codebook changes discarded and reloaded from saved version.")
    st.rerun()

# Save Changes
if col_save.button("Save Changes", key="accept_codebook_changes_main"):
    df_to_save_codebook = st.session_state.edited_codebook_df.drop(columns=["Select"], errors="ignore").reset_index(drop=True)
    if data_manager.save_codebook(df_to_save_codebook, st.session_state.project_path):
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
        # Update Codes
        existing_codes = robust_comma_string_to_list_p02(df_to_update.at[row_idx, 'Codes'])
        for code in code_names:
            if code not in existing_codes:
                existing_codes.append(code)
        df_to_update.at[row_idx, 'Codes'] = robust_list_to_comma_string_p02(existing_codes)
        # Update code_ids
        existing_code_ids = robust_comma_string_to_list_p02(df_to_update.at[row_idx, 'code_ids'])
        for code in code_names:
            meta = st.session_state.edited_codebook_df[
                st.session_state.edited_codebook_df["Code Name"] == code
            ]
            if not meta.empty:
                cid = str(meta.iloc[0]["code_id"])
                if cid not in existing_code_ids:
                    existing_code_ids.append(cid)
        df_to_update.at[row_idx, 'code_ids'] = robust_list_to_comma_string_p02(existing_code_ids)
    st.session_state.data_for_coding_tab3 = df_to_update
    selected_paths = st.session_state.get('currently_selected_view_filepaths_for_saving_tab3', [])
    if len(selected_paths) == 1:
        df_to_save = df_to_update.copy()
        if 'Source View' in df_to_save.columns:
            df_to_save = df_to_save.drop(columns=['Source View'], errors='ignore')
        success = data_manager.save_coded_data_to_view(df_to_save, selected_paths[0])
        if success:
            ui_helpers.show_success_message(f"Codes saved to '{os.path.basename(selected_paths[0])}'.")
        else:
            ui_helpers.show_error_message(f"Failed to save codes to '{os.path.basename(selected_paths[0])}'.")
    else:
        ui_helpers.show_info("Codes updated in-memory. Use 'Save All Row Codes' to persist to files.")
    st.rerun()



# Trigger AI codebook review dialog after generator closes
if st.session_state.pop("show_ai_codebook_review_dialog", False):
    ai_codebook_review_dialog()

st.divider()