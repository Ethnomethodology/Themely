# app.py
import streamlit as st
import pandas as pd # Keep for any helpers that might remain or for type hints
import os
import requests
import yaml
from datetime import datetime
import shutil
# import subprocess # Not strictly needed for ONNX GenAI direct Python usage
# import platform # Not strictly needed for ONNX GenAI direct Python usage
import zipfile # Keep for other potential uses
import io # Keep for other potential uses
# import sys # Not strictly needed now

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    layout="wide",
    page_title="Themely - Qualitative Analysis"
)

from modules import auth, utils, ui_helpers, db # Moved imports down

# --- Initialize Database and Global Settings ---
if 'db_initialized' not in st.session_state:
    db.init_db()
    st.session_state.db_initialized = True

if 'model_cache_dir' not in st.session_state:
    st.session_state.model_cache_dir = db.load_setting("model_cache_dir") or os.path.join(os.path.expanduser("~"), ".themely_models")
if 'ui_model_cache_dir_input' not in st.session_state: 
    st.session_state.ui_model_cache_dir_input = st.session_state.model_cache_dir

if 'past_projects' not in st.session_state: st.session_state.past_projects = db.load_project_index()
if 'removed_projects' not in st.session_state: st.session_state.removed_projects = []
st.session_state.past_projects = [p for p in st.session_state.past_projects if p['name'] not in st.session_state.removed_projects]

logger = utils.setup_logger("app_main_nav")

# --- Initialize Global Session States ---
if 'project_action_choice' not in st.session_state: st.session_state.project_action_choice = "New Project"
if 'current_project_name' not in st.session_state: st.session_state.current_project_name = None
if 'project_config' not in st.session_state: st.session_state.project_config = {}
if 'project_path' not in st.session_state: st.session_state.project_path = None

if 'ui_project_directory_input' not in st.session_state: st.session_state.ui_project_directory_input = os.path.expanduser("~")
if 'ui_project_name_input' not in st.session_state: st.session_state.ui_project_name_input = ""
if 'ui_open_config_file_path_input' not in st.session_state: st.session_state.ui_open_config_file_path_input = ""

if 'default_local_llm_for_codebook' not in st.session_state:
    st.session_state.default_local_llm_for_codebook = "Phi-4 Mini Instruct (ONNX)"

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

LOCAL_AI_METHODS_CONFIG = {
    "codebook_generation": {
        "default_method": "LLM-Assisted Code Name Suggestion", 
        "options": {
            "LLM-Assisted Code Name Suggestion": { 
                "description": "Uses a local LLM to suggest code names based on text content. Further refinement (descriptions, rationales) is done manually.",
                "llm_choices": {
                    "Phi-4 Mini Instruct (ONNX)": { # Corrected key to match intended usage
                        "engine": "onnxruntime-genai",
                        "model_id_hf": "microsoft/Phi-4-mini-instruct-onnx", 
                        "model_subfolder_hf": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4", 
                        "local_model_foldername": "Phi-4-mini-instruct-onnx-cpu-int4", 
                        "model_size": "~4.2B params (INT4 quantized, ~2.2 GB)", 
                        "min_requirements": "CPU (Modern, AVX2), ~4-5 GB RAM available",
                        "download_function_key": "download_phi3_model_task" 
                    },
                    "Gemma-3 1B (ONNX-GQA)": {
                        "engine": "onnxruntime-genai",
                        "model_id_hf": "onnx-community/gemma-3-1b-it-ONNX-GQA",
                        "model_subfolder_hf": "onnx",
                        "local_model_foldername": "gemma-3-1b-it-onnx-gqa",
                        "model_size": "~1B parameters (quantized 4-bit, ~0.8 GB)",
                        "min_requirements": "CPU (Modern, AVX2), ~2 GB RAM available",
                        "download_function_key": "download_gemma_model_task"
                    }
                }
            }
        }
    },
    "coding_with_codebook": { 
        "default_method": "Embedding Similarity Matching", 
        "options": {
            "Embedding Similarity Matching": {
                "libraries": ["SentenceTransformers (all-MiniLM-L6-v2)"],
                "model_size": "~80MB (MiniLM)",
                "min_requirements": "CPU (Modern), ~500MB RAM available",
                "description": "Generates embeddings for posts and codebook entries, then assigns codes based on cosine similarity above a threshold."
            },
            "LLM-Assisted Coding (Local)": {
                 "description": "Uses a selected local LLM to apply codes from the existing codebook to text items.",
                 "llm_choices": { 
                    "Phi-4 Mini Instruct (ONNX)": { # Corrected key
                        "engine": "onnxruntime-genai",
                        "model_id_hf": "microsoft/Phi-4-mini-instruct-onnx",
                        "model_subfolder_hf": "cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
                        "local_model_foldername": "Phi-4-mini-instruct-onnx-cpu-int4",
                        "model_size": "~4.2B params (INT4 quantized, ~2.2 GB)",
                        "min_requirements": "CPU (Modern, AVX2), ~4-5 GB RAM available",
                        "download_function_key": "download_phi3_model_task_apply" 
                    },
                    "Gemma-3 1B (ONNX-GQA)": {
                        "engine": "onnxruntime-genai",
                        "model_id_hf": "onnx-community/gemma-3-1b-it-ONNX-GQA",
                        "model_subfolder_hf": "onnx",
                        "local_model_foldername": "gemma-3-1b-it-onnx-gqa",
                        "model_size": "~1B parameters (quantized 4-bit, ~0.8 GB)",
                        "min_requirements": "CPU (Modern, AVX2), ~2 GB RAM available",
                        "download_function_key": "download_gemma_model_task_apply"
                    }
                 }
            }
        }
    },
    "grouping_codes_to_themes": { 
        "default_method": "Embedding-based Clustering",
        "options": {
            "Embedding-based Clustering": {"libraries": ["SentenceTransformers (all-MiniLM-L6-v2)", "Scikit-learn"],"model_size": "~80MB (MiniLM) + Scikit-learn (variable)","min_requirements": "CPU (Modern), ~500MB RAM available","description": "Embeds code definitions and clusters them using algorithms like Agglomerative Clustering or K-Means."}
        }
    },
    "generating_theme_summaries": { 
        "default_method": "Extractive Summarization with References",
        "options": {
            "Extractive Summarization with References": {"libraries": ["Sumy (LexRank/LSA) or Gensim (TextRank)"],"model_size": "Minimal (algorithm-based)","min_requirements": "CPU, ~200MB RAM available","description": "Selects salient sentences directly from the theme's posts and provides references (unique_app_id, code_ids) for each summarized part."}
        }
    }
}

def project_setup_page_content():
    st.title("Themely")
    st.header("Project Setup")
    st.write("Get started by creating a new project or opening an existing one, then enter your Reddit and AI service API keys or configure Local AI.")
    ui_helpers.page_sidebar_info(["Create or open a project","Enter Reddit API credentials","Configure AI provider (Cloud or Local)"])
    left_col, right_col = st.columns([1, 1])
    with left_col:
        project_action_key = "project_action_radio_home_nav"
        project_action_index = 0 if st.session_state.get('project_action_choice', "New Project") == "New Project" else 1
        project_action = st.radio("Action:", ("New Project", "Open Project"), index=project_action_index, key=project_action_key, horizontal=True, on_change=lambda: setattr(st.session_state, 'project_action_choice', st.session_state[project_action_key]))
        if st.session_state.project_action_choice == "New Project":
            st.markdown("Set project name and directory.")
            col_dir_cr, col_sep_cr_ui, col_name_cr = st.columns([0.7, 0.05, 0.25])
            with col_dir_cr: dir_input_key_cr = "ui_create_dir_input_key_home_nav"; ui_dir_val_cr = st.text_input("Directory:", value=st.session_state.ui_project_directory_input, key=dir_input_key_cr, on_change=lambda: setattr(st.session_state, 'ui_project_directory_input', st.session_state[dir_input_key_cr]))
            with col_sep_cr_ui: st.markdown(f"<div style='text-align: center; margin-top: 28px; font-size: 1.2em;'>/</div>", unsafe_allow_html=True)
            with col_name_cr: name_input_key_cr = "ui_create_name_input_key_home_nav"; ui_name_val_cr = st.text_input("Project name:", value=st.session_state.ui_project_name_input, key=name_input_key_cr, on_change=lambda: setattr(st.session_state, 'ui_project_name_input', st.session_state[name_input_key_cr]))
            if st.button("Create", key="create_project_button_action_home_nav"):
                dir_create, name_create = st.session_state.ui_project_directory_input, st.session_state.ui_project_name_input
                if not dir_create or not name_create: ui_helpers.show_error_message("Directory and Name are required.")
                elif not os.path.isabs(dir_create): ui_helpers.show_error_message("Directory must be an absolute path.")
                else:
                    proj_id = utils.generate_project_id(name_create); proj_folder = os.path.join(dir_create, proj_id)
                    try: os.makedirs(proj_folder, exist_ok=True)
                    except Exception as e: ui_helpers.show_error_message(f"Failed to create folder '{proj_folder}': {e}"); return
                    if auth.setup_project_storage(name_create, "Local", proj_folder):
                        st.session_state.current_project_name, st.session_state.project_path = name_create, proj_folder
                        cfg_file = utils.PROJECT_CONFIG_FILE_TEMPLATE.format(project_name_id=proj_id)
                        db.save_project_index(name_create, os.path.join(proj_folder, cfg_file))
                        st.session_state.past_projects = [p for p in db.load_project_index() if p['name'] not in st.session_state.removed_projects]
                        st.session_state.project_config = utils.load_project_config(proj_folder, proj_id)
                        if not st.session_state.project_config: ui_helpers.show_error_message("Critical: Config not loaded after creation."); st.session_state.current_project_name = None; st.session_state.project_path = None; return
                        st.session_state.project_config.update({'project_name': name_create, 'project_config_file_directory': proj_folder, 'storage_type': "Local", 'project_id_for_filename': proj_id})
                        utils.save_project_config(proj_folder, proj_id, st.session_state.project_config)
                        initialize_project_data_states(); ui_helpers.show_success_message(f"Project '{name_create}' created in '{proj_folder}'!"); st.rerun()
                    else: ui_helpers.show_error_message("Failed to create project. Check permissions/logs.")
        elif st.session_state.project_action_choice == "Open Project":
            open_cfg_key = "ui_open_config_file_path_input_key_home_nav"
            cfg_path_open = st.text_input("Project config file:", value=st.session_state.ui_open_config_file_path_input, key=open_cfg_key, on_change=lambda: setattr(st.session_state, 'ui_open_config_file_path_input', st.session_state[open_cfg_key]))
            if st.button("Open", key="open_project_button_action_home_nav"):
                if not cfg_path_open or not os.path.isabs(cfg_path_open) or not os.path.isfile(cfg_path_open): ui_helpers.show_error_message(f"Invalid path or file not found: {cfg_path_open}"); return
                try:
                    cfg_dir = os.path.dirname(cfg_path_open); base_name = os.path.basename(cfg_path_open)
                    proj_id_file = base_name.rsplit('_', 1)[0] if base_name.endswith(f"_{utils.PROJECT_CONFIG_FILE_TEMPLATE.split('_', 1)[1]}") else None
                    loaded_cfg = utils.load_project_config(cfg_dir, proj_id_file)
                    if not loaded_cfg: 
                        with open(cfg_path_open, 'r') as f: temp_cfg = yaml.safe_load(f)
                        proj_name_content = temp_cfg.get('project_name')
                        if not proj_name_content: ui_helpers.show_error_message("Name not in config. Cannot open by direct path if ID fails."); return
                        proj_id_name = utils.generate_project_id(proj_name_content)
                        loaded_cfg = utils.load_project_config(cfg_dir, proj_id_name)
                    if not loaded_cfg: ui_helpers.show_error_message(f"Error processing config. May be corrupted or name/ID mismatch."); return
                    st.session_state.project_config = loaded_cfg; st.session_state.current_project_name = loaded_cfg.get('project_name'); st.session_state.project_path = cfg_dir
                    db.save_project_index(st.session_state.current_project_name, cfg_path_open)
                    st.session_state.past_projects = [p for p in db.load_project_index() if p['name'] not in st.session_state.removed_projects]
                    st.session_state.project_config.update({'project_config_file_directory': cfg_dir, 'storage_type': st.session_state.project_config.get('storage_type', "Local")})
                    if 'project_id_for_filename' not in st.session_state.project_config: st.session_state.project_config['project_id_for_filename'] = utils.generate_project_id(st.session_state.current_project_name)
                    st.session_state.ui_project_name_input = st.session_state.current_project_name; st.session_state.ui_project_directory_input = os.path.dirname(st.session_state.project_path)
                    initialize_project_data_states(); st.rerun()
                except yaml.YAMLError: ui_helpers.show_error_message(f"Error parsing YAML config: {cfg_path_open}")
                except Exception as e: ui_helpers.show_error_message(f"Error opening project: {e}"); logger.error(f"Error opening project: {e}", exc_info=True)
    with right_col: 
        with st.container(height=300):
            st.subheader("Recent Projects")
            if not st.session_state.get('past_projects', []): st.info("No recent projects.")
            for proj in st.session_state.get('past_projects', []):
                is_active = proj['name'] == st.session_state.current_project_name; cfg_path_proj = os.path.normpath(proj['path']); base_path_proj = os.path.dirname(cfg_path_proj)
                color = "#d4edda" if is_active else "#fff3cd"; cols_recent = st.columns([0.7, 0.1, 0.1, 0.1], gap="small")
                with cols_recent[0]: st.markdown(f"<div style='background-color:{color};padding:2px;border-radius:4px;display:block;line-height:1.2;max-width:400px;'><strong style='color:#000'>{proj['name']}</strong><br><span style='color:#000;display:inline-block;width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;' title='{base_path_proj}'>{base_path_proj}</span></div>", unsafe_allow_html=True)
                with cols_recent[1]:
                    if st.button("", icon=":material/open_in_new:", key=f"open_past_{proj['name']}", help="Open this project"):
                        cfg_open_path, cfg_open_dir = proj['path'], os.path.dirname(proj['path']); proj_id_load = utils.generate_project_id(proj['name'])
                        loaded_cfg_open = utils.load_project_config(cfg_open_dir, proj_id_load)
                        if loaded_cfg_open: st.session_state.current_project_name, st.session_state.project_path, st.session_state.project_config = proj['name'], cfg_open_dir, loaded_cfg_open; initialize_project_data_states(); db.save_project_index(proj['name'], cfg_open_path); st.session_state.past_projects = [p for p in db.load_project_index() if p['name'] not in st.session_state.removed_projects]; st.rerun()
                        else: ui_helpers.show_error_message(f"Failed to load project '{proj['name']}' from {cfg_open_path}")
                with cols_recent[2]:
                    if st.button("", icon=":material/remove_circle:", key=f"remove_past_{proj['name']}", help="Remove from list (not delete files)"):
                        st.session_state.removed_projects.append(proj['name']); db.delete_project_index(proj['name']); st.session_state.past_projects = [p for p in st.session_state.past_projects if p['name'] not in st.session_state.removed_projects]
                        if st.session_state.current_project_name == proj['name']: st.session_state.current_project_name, st.session_state.project_path, st.session_state.project_config = None, None, {}
                        st.rerun()
                with cols_recent[3]:
                    del_key_proj = f"delete_past_{proj['name']}_{utils.generate_project_id(proj['name'])}"
                    if st.button("", icon=":material/delete:", key=del_key_proj, help="Delete project files and folder"): st.session_state.confirm_delete_project_name, st.session_state.confirm_delete_project_path_to_config = proj['name'], proj['path']; st.rerun()
    if st.session_state.get('confirm_delete_project_name'): 
        name_del, path_cfg_del = st.session_state.confirm_delete_project_name, st.session_state.confirm_delete_project_path_to_config; folder_del = os.path.dirname(path_cfg_del)
        st.error(f"**Confirm Deletion:** Permanently delete all files for project '{name_del}' at `{folder_del}`? This cannot be undone.")
        confirm_del_col1, confirm_del_col2, _ = st.columns([1,1,3])
        if confirm_del_col1.button("YES, DELETE", key="confirm_delete_files_btn_dialog", type="primary"):
            try:
                shutil.rmtree(folder_del); db.delete_project_index(name_del); st.session_state.removed_projects.append(name_del)
                st.session_state.past_projects = [p for p in st.session_state.past_projects if p['name'] != name_del]
                if st.session_state.current_project_name == name_del: st.session_state.current_project_name, st.session_state.project_path, st.session_state.project_config = None,None,{}
                ui_helpers.show_success_message(f"Project '{name_del}' and folder deleted.")
            except Exception as e: ui_helpers.show_error_message(f"Failed to delete project folder '{folder_del}': {e}")
            st.session_state.confirm_delete_project_name, st.session_state.confirm_delete_project_path_to_config = None, None; st.rerun()
        if confirm_del_col2.button("CANCEL", key="cancel_delete_files_btn_dialog"): st.session_state.confirm_delete_project_name, st.session_state.confirm_delete_project_path_to_config = None, None; st.rerun()
    if st.session_state.current_project_name: ui_helpers.nav_button("Edit API Keys or Go to Data Management", "pages/data_management.py", key="nav_to_data_mgmt_home")
    if st.session_state.current_project_name: 
        st.divider(); st.header("API Keys & Local AI Configuration")
        if st.session_state.project_path:
            proj_id_keys = st.session_state.project_config.get('project_id_for_filename') or utils.generate_project_id(st.session_state.current_project_name)
            api_cfg_disp = utils.load_project_config(st.session_state.project_path, proj_id_keys) or st.session_state.project_config
            with st.expander("Reddit API", expanded=False): 
                reddit_cfg = api_cfg_disp.get('reddit_api', {}); client_id = st.text_input("Client ID", type="password", value=reddit_cfg.get('client_id', ''), key="api_reddit_client_id_form_home_nav_exp")
                client_secret = st.text_input("Client Secret", type="password", value=reddit_cfg.get('client_secret', ''), key="api_reddit_client_secret_form_home_nav_exp")
                user_agent_def = f'Python:ThemelyApp_{proj_id_keys}:v0.1 by /u/YourUsername'; user_agent = st.text_input("User Agent", value=reddit_cfg.get('user_agent', user_agent_def), key="api_reddit_user_agent_form_home_nav_exp")
            with st.expander("AI Service Configuration", expanded=True): 
                ai_providers = ["OpenAI", "Gemini", "Local"]; ai_provider_def = api_cfg_disp.get('ai_provider', 'OpenAI')
                ai_provider_idx = ai_providers.index(ai_provider_def) if ai_provider_def in ai_providers else 0
                ai_provider_sel = st.selectbox("Provider", ai_providers, index=ai_provider_idx, key="api_ai_provider_select_form_home_nav_exp")
                ai_key_input, local_ai_settings = "", {}
                if ai_provider_sel in ["OpenAI", "Gemini"]:
                    ai_key_val = api_cfg_disp.get(f'{ai_provider_sel.lower()}_api', {}).get('api_key', '')
                    ai_key_input = st.text_input("API Key", type="password", value=ai_key_val, key=f"api_ai_key_input_form_home_nav_{ai_provider_sel}_exp")
                elif ai_provider_sel == "Local": 
                    st.markdown("---"); st.subheader("Local AI Setup"); st.write("Themely will use local models for AI tasks. No data is sent to external services.")
                    st.markdown("**Model Cache Directory**"); cache_col1, cache_col2 = st.columns([3,1])
                    with cache_col1: st.session_state.ui_model_cache_dir_input = st.text_input("Cache Path:", value=st.session_state.ui_model_cache_dir_input, key="global_model_cache_dir_input_app_exp")
                    with cache_col2:
                        st.write(""); st.write("") 
                        if st.button("Save Cache Path", key="save_global_model_cache_dir_btn_app_exp"):
                            new_path = st.session_state.ui_model_cache_dir_input.strip()
                            if os.path.isabs(new_path):
                                try: os.makedirs(new_path, exist_ok=True); db.save_setting("model_cache_dir", new_path); st.session_state.model_cache_dir = new_path; ui_helpers.show_success_message(f"Cache saved: {new_path}"); st.rerun()
                                except Exception as e: ui_helpers.show_error_message(f"Error setting cache: {e}")
                            else: ui_helpers.show_error_message("Cache directory must be an absolute path.")
                    st.markdown("---"); st.markdown("**Local AI Methods & Models:**"); current_local_cfg = api_cfg_disp.get('local_ai_config', {})
                    for task_key, task_val in LOCAL_AI_METHODS_CONFIG.items():
                        task_name_disp = task_key.replace("_", " ").title(); st.markdown(f"**{task_name_disp}:**")
                        method_key_ui = f"local_method_{task_key}_ui_exp"; sel_method = current_local_cfg.get(task_key, {}).get("method", task_val["default_method"])
                        if sel_method not in task_val["options"]: sel_method = task_val["default_method"]
                        method_opts = list(task_val["options"].keys()); sel_method_idx = method_opts.index(sel_method) if sel_method in method_opts else 0
                        chosen_method = st.selectbox(f"Method for {task_name_disp}:", options=method_opts, index=sel_method_idx, key=method_key_ui)
                        method_details_disp = task_val["options"][chosen_method]; local_ai_settings[task_key] = {"method": chosen_method} 
                        if "llm_choices" in method_details_disp: 
                            st.markdown("Select Local LLM:")
                            llm_choice_key_ui = f"local_llm_choice_{task_key}_ui_exp"
                            current_llm_for_task = current_local_cfg.get(task_key, {}).get("llm_model", st.session_state.default_local_llm_for_codebook)
                            llm_options_for_task = list(method_details_disp["llm_choices"].keys())
                            if current_llm_for_task not in llm_options_for_task: current_llm_for_task = st.session_state.default_local_llm_for_codebook
                            llm_choice_idx = llm_options_for_task.index(current_llm_for_task) if current_llm_for_task in llm_options_for_task else 0
                            chosen_llm = st.selectbox(f"LLM for {task_name_disp}:", options=llm_options_for_task, index=llm_choice_idx, key=llm_choice_key_ui)
                            local_ai_settings[task_key]["llm_model"] = chosen_llm 
                            llm_specific_details = method_details_disp["llm_choices"][chosen_llm]
                            st.markdown(f"*   **Engine:** `{llm_specific_details['engine']}`")
                            st.markdown(f"*   **Approx. Model Size(s):** `{llm_specific_details['model_size']}`")
                            st.markdown(f"*   **Min. System Requirements:** `{llm_specific_details['min_requirements']}`")
                            model_cache_path = st.session_state.model_cache_dir
                            if llm_specific_details['engine'] == "llama.cpp": 
                                gemma_folder = os.path.join(model_cache_path, llm_specific_details['local_model_foldername']); gemma_model_file = os.path.join(gemma_folder, llm_specific_details['model_filename_hf']) 
                                os.makedirs(gemma_folder, exist_ok=True) 
                                if os.path.exists(gemma_model_file): st.success(f"{chosen_llm} model found.")
                                else: 
                                    if st.button(f"Download {chosen_llm} Model", key=llm_specific_details['download_function_key']): download_gemma_model_dialog()
                            elif llm_specific_details['engine'] == "onnxruntime-genai": 
                                phi3_model_dir = os.path.join(model_cache_path, llm_specific_details['local_model_foldername'])
                                if os.path.isdir(phi3_model_dir) and len(os.listdir(phi3_model_dir)) > 0 : st.success(f"{chosen_llm} model files found in cache.")
                                else:
                                    if st.button(f"Download {chosen_llm} Model Files", key=llm_specific_details['download_function_key']): st.session_state.phi3_download_details = llm_specific_details; download_phi3_mini_model_dialog()
                        else: 
                            st.markdown(f"*   **Uses:** `{', '.join(method_details_disp['libraries'])}`"); st.markdown(f"*   **Approx. Model Size(s):** `{method_details_disp['model_size']}`"); st.markdown(f"*   **Min. System Requirements:** `{method_details_disp['min_requirements']}`")
                        st.markdown(f"*   *{method_details_disp['description']}*"); st.markdown("---")
                if st.button("Save Configuration", key="api_save_keys_button_form_home_nav_exp"):
                    reddit_keys_save = {"client_id": client_id, "client_secret": client_secret, "user_agent": user_agent}; ai_keys_save = {}
                    if ai_provider_sel in ["OpenAI", "Gemini"]: ai_keys_save = {"api_key": ai_key_input}
                    elif ai_provider_sel == "Local": ai_keys_save = local_ai_settings 
                    valid_r, valid_ai = all(reddit_keys_save.values()), True
                    if ai_provider_sel != "Local" and not ai_keys_save.get("api_key"): valid_ai = False
                    if not valid_r: ui_helpers.show_warning_message("Reddit API fields incomplete.")
                    elif not valid_ai: ui_helpers.show_warning_message(f"{ai_provider_sel} API Key empty.")
                    elif auth.validate_api_keys(reddit_keys_save, "Reddit") and auth.validate_api_keys(ai_keys_save if ai_provider_sel != "Local" else {}, ai_provider_sel):
                        proj_id_save_cfg = st.session_state.project_config.get('project_id_for_filename') or utils.generate_project_id(st.session_state.current_project_name)
                        if auth.store_api_keys(reddit_keys_save, ai_keys_save, ai_provider_sel):
                            st.session_state.project_config = utils.load_project_config(st.session_state.project_path, proj_id_save_cfg)
                            ui_helpers.show_success_message("Configuration saved to project."); st.rerun()

# --- DEBUGGING ADVICE ---
# The DeltaGenerator output you're seeing means a Streamlit object is being printed.
# Search your ENTIRE codebase (app.py and all .py files in the 'modules' folder)
# for any `print()` statements that might be accidentally printing the result of a
# Streamlit command (e.g., `print(st.button(...))` or `print(some_variable_assigned_from_st_call)`).
# Remove or comment out such print statements. The issue is most likely there.
# This output typically appears if the `__repr__` of a DeltaGenerator object is invoked.
# The sidebar rendering code below is standard and UNLIKELY to be the direct cause
# unless one of its return values is captured and printed elsewhere.

st.sidebar.title("Themely")
project_name_sidebar = st.session_state.get('current_project_name') or "None"
if project_name_sidebar != "None":
    # The return value of st.sidebar.success is a DeltaGenerator. DO NOT PRINT IT.
    _ = st.sidebar.success(f"Active project: {project_name_sidebar}") 
else:
    # The return value of st.sidebar.warning is a DeltaGenerator. DO NOT PRINT IT.
    _ = st.sidebar.warning("Active project: None") 
st.sidebar.divider()
# The return value of st.sidebar.info is a DeltaGenerator. DO NOT PRINT IT.
_ = st.sidebar.info(f"Model Cache: {st.session_state.get('model_cache_dir', 'Not Set')}")


@st.dialog("Download Gemma Model", width="medium")
def download_gemma_model_dialog():
    st.markdown("#### Download Gemma‑3‑1B‑pt QAT (Q4_0)")
    st.markdown("""1. Log in to **Hugging Face** and accept the Gemma licence: <https://huggingface.co/google/gemma-3-1b-pt-qat-q4_0-gguf>.\n2. Create a *read‑only* access token (Settings → **Access Tokens**) and paste it below.\nThe script will fetch the quantised GGUF into your local model cache.""")
    cache_dir_gemma = st.session_state.model_cache_dir; gemma_details = None
    for task_cfg in LOCAL_AI_METHODS_CONFIG.values():
        if "llm_choices" in task_cfg["options"].get("LLM-Assisted Code Name Suggestion", {}): gemma_details = task_cfg["options"]["LLM-Assisted Code Name Suggestion"]["llm_choices"].get("Gemma-3 1B (GGUF)"); break
        elif "llm_choices" in task_cfg["options"].get("LLM-Assisted Coding (Local)", {}): gemma_details = task_cfg["options"]["LLM-Assisted Coding (Local)"]["llm_choices"].get("Gemma-3 1B (GGUF)"); break
    if not gemma_details: st.error("Gemma model config not found."); st.button("Close"); return
    gemma_folder_name, gemma_model_filename, gemma_hf_repo_id = gemma_details['local_model_foldername'], gemma_details['model_filename_hf'], gemma_details['model_id_hf']
    target_gemma_folder, target_gemma_model_path = os.path.join(cache_dir_gemma, gemma_folder_name), os.path.join(target_gemma_folder, gemma_model_filename)
    token_gemma = st.text_input("Hugging Face Token (optional, for gated models)", type="password", key="hf_token_input_gemma")
    if os.path.isdir(target_gemma_folder) and any(f.endswith('.onnx') for f in os.listdir(target_gemma_folder)):
        st.success("Gemma ONNX model already present.")
    else:
        if st.button("Download Gemma Model Files", key="hf_start_download_gemma"):
            os.makedirs(target_gemma_folder, exist_ok=True)
            st.info("Downloading Gemma-3-1B ONNX model...")
            from huggingface_hub import snapshot_download
            temp_download_path = os.path.join(target_gemma_folder, "_temp_gemma_download")
            os.makedirs(temp_download_path, exist_ok=True)
            try:
                with st.spinner("Fetching Gemma ONNX files..."):
                    snapshot_download(
                        repo_id=gemma_hf_repo_id,
                        allow_patterns=[f"{gemma_details['model_subfolder_hf']}/*"],
                        local_dir=temp_download_path,
                        local_dir_use_symlinks=False,
                        token=token_gemma if token_gemma else None
                    )
                actual_model_files_path = os.path.join(temp_download_path, gemma_details['model_subfolder_hf'])
                if os.path.isdir(actual_model_files_path):
                    for item_name in os.listdir(actual_model_files_path):
                        s_item = os.path.join(actual_model_files_path, item_name)
                        d_item = os.path.join(target_gemma_folder, item_name)
                        if os.path.isdir(s_item):
                            shutil.copytree(s_item, d_item, dirs_exist_ok=True)
                        else:
                            shutil.copy2(s_item, d_item)
                    shutil.rmtree(temp_download_path)
                    st.success("Gemma ONNX model downloaded successfully!"); st.rerun()
                else:
                    st.error(f"Could not find downloaded Gemma ONNX files at expected subpath: {actual_model_files_path}")
                    shutil.rmtree(temp_download_path)
            except Exception as e:
                st.error(f"Gemma ONNX download failed: {e}")
                if os.path.isdir(temp_download_path): shutil.rmtree(temp_download_path)
    if st.button("Cancel", key="hf_cancel_gemma"): st.rerun()

@st.dialog("Download Phi-4 Mini Model (ONNX)", width="medium")
def download_phi3_mini_model_dialog():
    st.markdown("#### Download Phi-4 Mini Instruct (ONNX CPU INT4)")
    phi3_details = st.session_state.get("phi3_download_details")
    if not phi3_details: st.error("Phi-4 model details not found."); st.button("Close"); return
    st.markdown(f"""This will download ONNX files for **{phi3_details['model_id_hf']} ({phi3_details['model_subfolder_hf']})**.""")
    cache_dir_phi3, target_phi3_model_dir = st.session_state.model_cache_dir, os.path.join(st.session_state.model_cache_dir, phi3_details['local_model_foldername'])
    token_phi3 = st.text_input("Hugging Face Token (optional)", type="password", key="hf_token_input_phi3")
    if os.path.isdir(target_phi3_model_dir) and any(f.endswith('.onnx') for f in os.listdir(target_phi3_model_dir)): st.success("Phi-4 Mini model files seem present.")
    else:
        if st.button("Download Phi-4 Mini Model Files", key="hf_start_download_phi3"):
            os.makedirs(target_phi3_model_dir, exist_ok=True); st.info("Downloading Phi-4 Mini...")
            from huggingface_hub import snapshot_download # Import here to avoid top-level import if not used
            try:
                with st.spinner("Downloading from Hugging Face Hub..."):
                    # snapshot_download will download into target_phi3_model_dir, and if model_subfolder_hf is part of the repo structure,
                    # it will create that structure within target_phi3_model_dir.
                    # We need the final path to point to the directory containing model.onnx, config.json etc.
                    # The local_model_foldername should be the name of this final directory.
                    # Example: if repo has 'cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/model.onnx'
                    # and local_model_foldername is 'Phi-4-mini-instruct-onnx-cpu-int4'
                    # We want cache_dir/Phi-4-mini-instruct-onnx-cpu-int4/model.onnx
                    
                    # Download all matching files into a temporary sub-directory within the target_phi3_model_dir first.
                    temp_download_path = os.path.join(target_phi3_model_dir, "_temp_phi3_download")
                    os.makedirs(temp_download_path, exist_ok=True)

                    snapshot_download(repo_id=phi3_details['model_id_hf'], allow_patterns=[f"{phi3_details['model_subfolder_hf']}/*"], local_dir=temp_download_path, local_dir_use_symlinks=False, token=token_phi3 if token_phi3 else None)
                    
                    # Now, find the actual model files within the downloaded structure
                    # The files we want are inside temp_download_path/phi3_details['model_subfolder_hf']
                    actual_model_files_path = os.path.join(temp_download_path, phi3_details['model_subfolder_hf'])
                    
                    if os.path.isdir(actual_model_files_path):
                        logger.info(f"Phi-4 files downloaded to {actual_model_files_path}. Moving to target: {target_phi3_model_dir}")
                        for item_name in os.listdir(actual_model_files_path):
                            s_item = os.path.join(actual_model_files_path, item_name)
                            d_item = os.path.join(target_phi3_model_dir, item_name) # Move directly into target_phi3_model_dir
                            if os.path.isdir(s_item): shutil.copytree(s_item, d_item, dirs_exist_ok=True)
                            else: shutil.copy2(s_item, d_item)
                        shutil.rmtree(temp_download_path) # Clean up temp download folder
                        logger.info("Phi-4 files moved successfully.")
                        st.success("Phi-4 Mini model files downloaded successfully!"); st.rerun()
                    else:
                        st.error(f"Could not find downloaded Phi-4 model files at expected subpath: {actual_model_files_path}")
                        logger.error(f"Expected Phi-4 path not found after snapshot_download: {actual_model_files_path}")
                        shutil.rmtree(temp_download_path) # Clean up

            except Exception as e_phi3_dl: st.error(f"Phi-4 Mini download failed: {e_phi3_dl}"); logger.error(f"Phi-4 Mini download error: {e_phi3_dl}", exc_info=True)
    if st.button("Cancel", key="hf_cancel_phi3"): st.rerun()

page_definitions = [st.Page(project_setup_page_content, title="Project Setup", default=True, icon="🛠️"), st.Page("pages/data_management.py", title="Data Management", icon="💾"), st.Page("pages/codebook.py", title="Codebook", icon="📖"), st.Page("pages/themes.py", title="Themes", icon="📊"), st.Page("pages/analysis.py", title="Analysis", icon="📝")]
pg = st.navigation(page_definitions)
# It's critical that pg.run() is called.
# Any accidental `print(pg)` or `print(pg.run())` could also cause issues.
# The return value of pg.run() is not typically used.
pg.run() 

if st.session_state.current_project_name:
    proj_id_rec = st.session_state.project_config.get('project_id_for_filename') or utils.generate_project_id(st.session_state.current_project_name)
    if not st.session_state.project_path and st.session_state.project_config.get('project_config_file_directory'):
        st.session_state.project_path = st.session_state.project_config['project_config_file_directory']
    elif st.session_state.project_path and st.session_state.project_config.get('project_config_file_directory') != st.session_state.project_path:
        loaded_cfg_rec = utils.load_project_config(st.session_state.project_path, proj_id_rec)
        if loaded_cfg_rec:
            st.session_state.project_config = loaded_cfg_rec
            st.session_state.project_config.update({'project_config_file_directory': st.session_state.project_path, 'storage_type': st.session_state.project_config.get('storage_type', "Local")})
            if 'project_id_for_filename' not in st.session_state.project_config: st.session_state.project_config['project_id_for_filename'] = proj_id_rec
            # The return value of save_project_config is boolean, not a DeltaGenerator. So this line is fine.
            utils.save_project_config(st.session_state.project_path, proj_id_rec, st.session_state.project_config)
        else: logger.error(f"Recovery failed for {st.session_state.current_project_name} from project_path.")

# Final check: Ensure no other code after this point accidentally prints a Streamlit object.