# pages/ai_coding_page.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime
from modules import data_management, ui_helpers, utils, ai_services

logger = utils.setup_logger("p02_ai_coding")

st.title("AI Coding on Views")

if not st.session_state.get('current_project_name'):
    st.warning("👈 Please create or open a project first from the '🏠 Project Setup' page.")
    st.stop()

ai_provider_configured = st.session_state.project_config.get('ai_provider') and \
                         st.session_state.project_config.get(f"{st.session_state.project_config.get('ai_provider', '').lower()}_api", {}).get('api_key')

if not ai_provider_configured:
    st.warning("AI Provider and/or API Key not configured. Please set them on the '🏠 Project Setup' page to enable AI coding features in the 'AI Coding' tab.")

# Helper functions
def robust_list_to_comma_string_p02(code_input):
    if isinstance(code_input, list):
        return ", ".join(str(c).strip() for c in code_input if str(c).strip())
    if pd.isna(code_input) or not str(code_input).strip(): return ""
    return str(code_input).strip()

def robust_comma_string_to_list_p02(code_str):
    if pd.isna(code_str) or not str(code_str).strip(): return []
    if isinstance(code_str, list): 
        return [str(c).strip() for c in code_str if str(c).strip()]
    return [c.strip() for c in str(code_str).split(',') if c.strip()]

def load_and_combine_selected_views_for_coding_p02():
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
        id_cols_present = [col for col in ['unique_app_id', 'id'] if col in concatenated_df.columns]
        if id_cols_present:
            dedup_col = 'unique_app_id' if 'unique_app_id' in id_cols_present else id_cols_present[0]
            concatenated_df.drop_duplicates(subset=[dedup_col], keep='first', inplace=True)
        else: 
            concatenated_df.drop_duplicates(keep='first', inplace=True)
            logger.warning("No 'unique_app_id' or 'id' column found for deduplication during view combination for coding.")

        if 'Codes' not in concatenated_df.columns:
            concatenated_df['Codes'] = "" 
        else:
            concatenated_df['Codes'] = concatenated_df['Codes'].apply(robust_list_to_comma_string_p02)
        st.session_state.data_for_coding_tab3 = concatenated_df
    else:
        st.session_state.data_for_coding_tab3 = pd.DataFrame()

# --- UI SECTION 1: View Selection ---
st.subheader("Select Project View(s) for Coding")
available_views_meta_p02 = data_management.list_created_views_metadata()

if not available_views_meta_p02:
    st.info("No project views created yet. Go to '💾 Data Management' page to create a view.")
else:
    display_views_for_editor_p02 = []
    for view_meta_item_p02 in available_views_meta_p02:
        view_key_p02 = view_meta_item_p02["view_name"] 
        if view_key_p02 not in st.session_state.selected_project_views_info_tab3: 
            st.session_state.selected_project_views_info_tab3[view_key_p02] = {"selected": False, "metadata": view_meta_item_p02}
        
        display_views_for_editor_p02.append({
            "Select": st.session_state.selected_project_views_info_tab3[view_key_p02].get("selected", False),
            "View Name": view_meta_item_p02["view_name"],
            "Created On": datetime.fromisoformat(view_meta_item_p02.get("creation_timestamp", "")).strftime("%Y-%m-%d %H:%M") if view_meta_item_p02.get("creation_timestamp") else "N/A",
            "Source Files Info": str(view_meta_item_p02.get("source_files_info", "N/A")) 
        })
    
    views_df_for_editor_p02 = pd.DataFrame(display_views_for_editor_p02)
    edited_views_df_for_selection_p02 = st.data_editor(
        views_df_for_editor_p02,
        column_config={"Select": st.column_config.CheckboxColumn(required=True)},
        disabled=[col for col in views_df_for_editor_p02.columns if col != "Select"],
        key="views_selector_editor_p02", hide_index=True,
        height=min(300, (len(views_df_for_editor_p02) + 1) * 35 + 3)
    )

    if not views_df_for_editor_p02.equals(edited_views_df_for_selection_p02):
        for idx_view_editor_p02, editor_row_view_selected_p02 in edited_views_df_for_selection_p02.iterrows():
            view_name_from_editor_p02 = editor_row_view_selected_p02["View Name"]
            if view_name_from_editor_p02 in st.session_state.selected_project_views_info_tab3:
                st.session_state.selected_project_views_info_tab3[view_name_from_editor_p02]["selected"] = editor_row_view_selected_p02["Select"]
        load_and_combine_selected_views_for_coding_p02() 
        st.rerun()

st.divider()

# --- UI SECTION 2: Data Display and Coding Actions ---
st.subheader("Data for Coding")
df_for_coding_master_p02 = st.session_state.get('data_for_coding_tab3', pd.DataFrame()) 

if df_for_coding_master_p02.empty:
    st.info("Select one or more views from the table above to load data for coding.")
else:
    search_coding_data_input_key_p02 = "search_coding_data_input_p02"
    search_coding_data_val_p02 = st.text_input("Search/Filter Coding Data (applies to display and manual bulk add):", 
                                                value=st.session_state.search_term_coding_data_tab3,
                                                key=search_coding_data_input_key_p02,
                                                on_change=lambda: setattr(st.session_state, 'search_term_coding_data_tab3', st.session_state[search_coding_data_input_key_p02]))
    st.session_state.search_term_coding_data_tab3 = search_coding_data_val_p02

    df_display_for_coding_editor_p02 = df_for_coding_master_p02.copy()
    if search_coding_data_val_p02:
            try:
                df_display_for_coding_editor_p02 = df_display_for_coding_editor_p02[
                    df_display_for_coding_editor_p02.astype(str).apply(
                        lambda r: r.str.contains(search_coding_data_val_p02, case=False, na=False, regex=True).any(), axis=1
                    )
                ]
            except Exception as e_search:
                st.warning(f"Error during search: {e_search}. Try a simpler search term.")
    
    st.caption("You can directly edit the 'Codes' column in the table below for individual item coding. Use the tabs further down for AI-assisted or bulk manual coding.")
    
    id_col_for_edit_and_ai = "unique_app_id"
    if id_col_for_edit_and_ai not in df_display_for_coding_editor_p02.columns:
        if "id" in df_display_for_coding_editor_p02.columns:
            id_col_for_edit_and_ai = "id"
        else: 
            id_col_for_edit_and_ai = None
            if search_coding_data_val_p02: 
                 st.warning("No 'unique_app_id' or 'id' column found in the filtered data. Direct edits in the table below might not be saved correctly if the search is active. Clear the search to ensure edits are saved.")

    edited_coding_df_from_editor_p02 = st.data_editor(
        df_display_for_coding_editor_p02, 
        column_config={ "Codes": st.column_config.TextColumn("Codes", help="Enter codes separated by commas") },
        key="coding_data_editor_main_p02", num_rows="dynamic", height=400, use_container_width=True,
        disabled=(search_coding_data_val_p02 and not id_col_for_edit_and_ai) 
    )
    
    if not df_display_for_coding_editor_p02.equals(edited_coding_df_from_editor_p02):
        if not id_col_for_edit_and_ai:
            st.warning("Cannot save direct table edits as no unique ID column (unique_app_id or id) is available in the displayed data.")
        else:
            edited_coding_df_from_editor_p02[id_col_for_edit_and_ai] = edited_coding_df_from_editor_p02[id_col_for_edit_and_ai].astype(str)
            current_master_df_for_update = st.session_state.data_for_coding_tab3.copy()
            current_master_df_for_update[id_col_for_edit_and_ai] = current_master_df_for_update[id_col_for_edit_and_ai].astype(str)
            update_dict_direct_edits = edited_coding_df_from_editor_p02.set_index(id_col_for_edit_and_ai)['Codes'].to_dict()
            current_master_df_for_update = current_master_df_for_update.set_index(id_col_for_edit_and_ai)
            for uid_direct_edit, new_codes_direct_edit in update_dict_direct_edits.items():
                if uid_direct_edit in current_master_df_for_update.index: 
                    current_master_df_for_update.loc[uid_direct_edit, 'Codes'] = new_codes_direct_edit
            st.session_state.data_for_coding_tab3 = current_master_df_for_update.reset_index()
            st.rerun()

    # --- Save Button Section (Moved Here) ---
    if not df_for_coding_master_p02.empty:
        col_spacer_save, col_save_button = st.columns([0.85, 0.15]) # Adjust ratio for right alignment
        with col_save_button:
            save_button_disabled = st.session_state.data_for_coding_tab3.empty
            if st.button("Save Coded Data", key="save_coded_data_p02_upper", disabled=save_button_disabled, use_container_width=True):
                df_to_save_p02 = st.session_state.data_for_coding_tab3.copy()
                selected_paths_p02 = st.session_state.get('currently_selected_view_filepaths_for_saving_tab3', [])
                
                if "Source View" in df_to_save_p02.columns:
                    df_to_save_p02 = df_to_save_p02.drop(columns=['Source View'], errors='ignore')

                if len(selected_paths_p02) == 1 and not st.session_state.get("prompt_save_combined_as_new_view_tab3", False):
                    if data_management.save_coded_data_to_view(df_to_save_p02, selected_paths_p02[0]):
                        ui_helpers.show_success_message(f"Codes saved successfully to existing view: '{os.path.basename(selected_paths_p02[0])}'.")
                        load_and_combine_selected_views_for_coding_p02() 
                        st.rerun()
                else: 
                    st.session_state.prompt_save_combined_as_new_view_tab3 = True
                    # No immediate rerun here, let the input field show below if needed
        
        # Logic for "Save as New View" if prompted
        if st.session_state.get("prompt_save_combined_as_new_view_tab3", False):
            st.info("Your coded data is derived from multiple views or you chose to save as new. Please provide a name for this new consolidated and coded view.")
            
            # Input and button for new view name
            col_new_view_name_input, col_new_view_save_button = st.columns([0.7, 0.3])
            with col_new_view_name_input:
                new_view_name_p02 = st.text_input("Enter Name for New Coded View:", key="new_coded_view_name_p02_input_upper")
            with col_new_view_save_button:
                st.write("") # Spacer for alignment
                if st.button("Confirm Save as New View", key="save_as_new_coded_view_p02_action_upper", use_container_width=True):
                    if not new_view_name_p02.strip(): 
                        ui_helpers.show_error_message("New view name cannot be empty.")
                    else:
                        source_view_names_p02 = [info["metadata"]["view_name"] for _, info in st.session_state.selected_project_views_info_tab3.items() if info.get("selected", False)]
                        if not source_view_names_p02: source_view_names_p02 = ["Combined/Edited Data from multiple sources"]
                        
                        df_save_new_p02 = st.session_state.data_for_coding_tab3.copy()
                        if "Source View" in df_save_new_p02.columns: 
                            df_save_new_p02 = df_save_new_p02.drop(columns=['Source View'], errors='ignore')

                        if data_management.save_project_view(df_save_new_p02, new_view_name_p02, source_filenames_info=source_view_names_p02): 
                            ui_helpers.show_success_message(f"New coded view '{new_view_name_p02}' saved successfully.")
                            st.session_state.prompt_save_combined_as_new_view_tab3 = False
                            st.session_state.selected_project_views_info_tab3 = {} 
                            st.session_state.data_for_coding_tab3 = pd.DataFrame()
                            st.session_state.currently_selected_view_filepaths_for_saving_tab3 = []
                            st.session_state.search_term_coding_data_tab3 = "" 
                            st.rerun()
    st.divider()

    # --- Tabs for AI and Manual Bulk Coding ---
    if not df_for_coding_master_p02.empty: # Only show tabs if there's data
        tab_ai_coding, tab_manual_coding = st.tabs(["AI Coding", "Manual Coding"])

        with tab_ai_coding:
            st.header("AI Code Generation")
            if not ai_provider_configured:
                st.info("AI features are disabled. Configure AI Provider and API Key in '🏠 Project Setup'.")
            else:
                ai_text_col_options_batch_p02 = [col for col in df_for_coding_master_p02.columns if df_for_coding_master_p02[col].dtype == 'object' and col not in ['Source View', 'Codes', 'unique_app_id', 'id']]
                if not ai_text_col_options_batch_p02 and not df_for_coding_master_p02.empty: 
                    st.warning("No suitable text columns (string type) found in the data for AI coding.")
                elif not df_for_coding_master_p02.empty and id_col_for_edit_and_ai: 
                    default_text_col_idx_batch_p02 = ai_text_col_options_batch_p02.index('text') if 'text' in ai_text_col_options_batch_p02 \
                                                  else (ai_text_col_options_batch_p02.index('title') if 'title' in ai_text_col_options_batch_p02 else 0)
                    
                    st.session_state.col_for_coding_tab3 = st.selectbox( 
                        "Select text column for AI input:", ai_text_col_options_batch_p02, 
                        index=default_text_col_idx_batch_p02, key="ai_text_col_select_p02_tab"
                    )
                    
                    id_col_for_ai_batch_p02 = id_col_for_edit_and_ai 
                    
                    st.caption(f"Using '`{id_col_for_ai_batch_p02}`' as the unique ID for matching AI results back to your data.")
                    ai_provider_batch_p02 = st.session_state.project_config.get('ai_provider', "OpenAI")
                    ai_model_batch_p02 = None
                    # Updated Model Names
                    if ai_provider_batch_p02 == "OpenAI": ai_model_batch_p02 = st.selectbox("OpenAI Model:", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], key="openai_model_coding_p02_tab")
                    elif ai_provider_batch_p02 == "Gemini": ai_model_batch_p02 = st.selectbox("Gemini Model:", ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.0-pro"], key="gemini_model_coding_p02_tab") 
                    
                    batch_prompt_template_p02 = st.text_area( "AI Batch Coding Prompt Template:", value=st.session_state.get("ai_batch_prompt_template_p02", """..."""), 
                                                              key="ai_batch_prompt_template_p02_input_tab", height=350,
                                                              on_change=lambda: st.session_state.update({"ai_batch_prompt_template_p02": st.session_state.ai_batch_prompt_template_p02_input_tab}))
                    
                    st.session_state.ai_batch_size_tab3 = st.number_input("Batch Size for AI Processing (items per request):", 1, 50, st.session_state.ai_batch_size_tab3, 1, key="batch_size_p02_tab")

                    if st.button("Generate Codes with AI (Batch Processing)", key="generate_ai_codes_batch_p02_tab", disabled=(not id_col_for_ai_batch_p02 or not ai_provider_configured)):
                        text_col_for_ai_run_p02 = st.session_state.col_for_coding_tab3
                        current_prompt_template_p02 = st.session_state.get("ai_batch_prompt_template_p02", batch_prompt_template_p02)
                        placeholders_ok = ('{json_data_batch}' in current_prompt_template_p02 and 
                                           '{text_column_name}' in current_prompt_template_p02 and
                                           '{id_column_name}' in current_prompt_template_p02)
                        if not text_col_for_ai_run_p02 or not ai_model_batch_p02 or not placeholders_ok:
                            ui_helpers.show_error_message("Ensure text column, AI model are selected, and prompt template includes {json_data_batch}, {text_column_name}, and {id_column_name} placeholders.")
                        elif st.session_state.data_for_coding_tab3.empty: 
                            ui_helpers.show_error_message("No data loaded in the 'Data for Coding' table.")
                        else:
                            df_to_code_run_p02 = st.session_state.data_for_coding_tab3.copy() 
                            if text_col_for_ai_run_p02 not in df_to_code_run_p02.columns or id_col_for_ai_batch_p02 not in df_to_code_run_p02.columns:
                                ui_helpers.show_error_message(f"Required columns for AI coding are missing: Text column '{text_col_for_ai_run_p02}' or ID column '{id_col_for_ai_batch_p02}'.")
                            else:
                                num_items_total_p02 = len(df_to_code_run_p02)
                                num_batches_run_p02 = (num_items_total_p02 - 1) // st.session_state.ai_batch_size_tab3 + 1
                                progress_bar_ai_run_p02 = st.progress(0, text=f"Initializing AI batch processing (0/{num_batches_run_p02})...")
                                all_ai_results_run_p02 = []
                                critical_batch_failure = False
                                prompt_template_for_batch_run_p02 = current_prompt_template_p02.replace("{text_column_name}", text_col_for_ai_run_p02).replace("{id_column_name}", id_col_for_ai_batch_p02)

                                with st.spinner(f"AI is processing {num_items_total_p02} item(s) in {num_batches_run_p02} batch(es)..."):
                                    for i_batch_p02 in range(num_batches_run_p02):
                                        batch_start_p02, batch_end_p02 = i_batch_p02 * st.session_state.ai_batch_size_tab3, (i_batch_p02 + 1) * st.session_state.ai_batch_size_tab3
                                        current_batch_df_p02 = df_to_code_run_p02.iloc[batch_start_p02:batch_end_p02]
                                        progress_bar_ai_run_p02.progress((i_batch_p02) / num_batches_run_p02, text=f"Sending Batch {i_batch_p02+1}/{num_batches_run_p02}...")
                                        batch_output_p02 = ai_services.generate_codes_for_batch_with_ai(current_batch_df_p02, ai_provider_batch_p02, prompt_template_for_batch_run_p02, text_col_for_ai_run_p02, id_col_for_ai_batch_p02, ai_model_batch_p02)
                                        if batch_output_p02 is None or not isinstance(batch_output_p02, list): 
                                            ui_helpers.show_error_message(f"AI processing for batch {i_batch_p02+1} failed critically. Aborting."); critical_batch_failure = True; break
                                        all_ai_results_run_p02.extend(batch_output_p02)
                                        progress_bar_ai_run_p02.progress((i_batch_p02 + 1) / num_batches_run_p02, text=f"Batch {i_batch_p02+1}/{num_batches_run_p02} processed.")
                                progress_bar_ai_run_p02.empty() 

                                if not critical_batch_failure:
                                    if not all_ai_results_run_p02 and num_items_total_p02 > 0 : ui_helpers.show_warning_message("AI batch coding returned no results.")
                                    elif num_items_total_p02 == 0: ui_helpers.show_info_message("No data for AI coding.")
                                    else:
                                        ai_results_df_p02 = pd.DataFrame(all_ai_results_run_p02)
                                        if 'unique_app_id' in ai_results_df_p02.columns and id_col_for_ai_batch_p02 != 'unique_app_id':
                                            ai_results_df_p02 = ai_results_df_p02.rename(columns={'unique_app_id': id_col_for_ai_batch_p02})
                                        
                                        expected_ids = set(df_to_code_run_p02[id_col_for_ai_batch_p02].astype(str))
                                        returned_ids_from_ai = set(ai_results_df_p02[id_col_for_ai_batch_p02].astype(str)) if id_col_for_ai_batch_p02 in ai_results_df_p02 else set()
                                        missing_ids = expected_ids - returned_ids_from_ai
                                        if missing_ids: ui_helpers.show_warning_message(f"{len(missing_ids)} items not in AI response. Logs: {list(missing_ids)[:3]}")
                                        
                                        df_to_update_p02 = st.session_state.data_for_coding_tab3.copy() 
                                        df_to_update_p02[id_col_for_ai_batch_p02] = df_to_update_p02[id_col_for_ai_batch_p02].astype(str)
                                        if id_col_for_ai_batch_p02 not in ai_results_df_p02.columns: 
                                            ui_helpers.show_error_message(f"AI response missing ID column '{id_col_for_ai_batch_p02}'. Cannot merge."); st.stop()
                                        ai_results_df_p02[id_col_for_ai_batch_p02] = ai_results_df_p02[id_col_for_ai_batch_p02].astype(str)
                                        ai_results_df_p02 = ai_results_df_p02.rename(columns={'Codes': 'AI_Generated_Codes'}, errors='ignore')
                                        if 'AI_Generated_Codes' not in ai_results_df_p02: ai_results_df_p02['AI_Generated_Codes'] = ""
                                        if 'error' not in ai_results_df_p02: ai_results_df_p02['error'] = None

                                        merged_df_p02 = pd.merge(df_to_update_p02, ai_results_df_p02[[id_col_for_ai_batch_p02, 'AI_Generated_Codes', 'error']], on=id_col_for_ai_batch_p02, how='left')
                                        coded_items, error_items = 0,0
                                        error_samples = []
                                        for idx, row in merged_df_p02.iterrows():
                                            has_err = pd.notna(row['error']) and str(row['error']).strip()
                                            if has_err: 
                                                error_items+=1
                                                if len(error_samples) < 3: error_samples.append(f"ID {row[id_col_for_ai_batch_p02]}: {str(row['error'])[:100]}...")
                                                logger.error(f"AI error for {row[id_col_for_ai_batch_p02]}: {row['error']}")
                                            if pd.notna(row['AI_Generated_Codes']) and str(row['AI_Generated_Codes']).strip():
                                                existing = robust_comma_string_to_list_p02(row['Codes'])
                                                new_ai = robust_comma_string_to_list_p02(row['AI_Generated_Codes'])
                                                merged_df_p02.loc[idx, 'Codes'] = robust_list_to_comma_string_p02(list(dict.fromkeys(existing + new_ai)))
                                                if not has_err: coded_items+=1
                                        st.session_state.data_for_coding_tab3 = merged_df_p02.drop(columns=['AI_Generated_Codes', 'error'], errors='ignore')
                                        if coded_items > 0: ui_helpers.show_success_message(f"AI coding done. {coded_items} items updated.")
                                        elif error_items == 0 and not missing_ids: ui_helpers.show_info_message("AI ran. No new codes generated or no errors.")
                                        if error_items > 0: ui_helpers.show_error_message(f"{error_items} items had AI errors. Samples: {'; '.join(error_samples) if error_samples else 'See logs.'}")
                                        st.rerun()
                elif not df_for_coding_master_p02.empty and not id_col_for_edit_and_ai:
                     st.error("Cannot perform AI Coding as a unique ID column ('unique_app_id' or 'id') is missing from the loaded data.")

        with tab_manual_coding:
            st.header("Manual Bulk Code Addition")
            st.caption("Enter a single code/phrase. This code will be added to all currently **displayed** rows in the 'Data for Coding' table above (respecting any active search/filter).")
            
            manual_code_input_key = "manual_code_input_p02_tab"
            manual_code_to_add = st.text_input("Code to add (do not use commas here):", key=manual_code_input_key,
                                               help="Enter a single code or thematic phrase. Commas are not allowed in this field.")

            disable_manual_add = df_for_coding_master_p02.empty or not id_col_for_edit_and_ai
            
            if st.button("Add Code to Displayed Rows", key="add_manual_code_button_p02_tab", disabled=disable_manual_add):
                code_val = st.session_state[manual_code_input_key].strip()
                if not code_val:
                    ui_helpers.show_error_message("Please enter a code to add.")
                elif ',' in code_val:
                    ui_helpers.show_error_message("Commas are not allowed in the code input field. Please enter a single code or phrase.")
                else:
                    if df_display_for_coding_editor_p02.empty: # df_display reflects the search filter
                        ui_helpers.show_warning_message("No rows are currently displayed (or match filter) to add codes to.")
                    else:
                        ids_to_update = df_display_for_coding_editor_p02[id_col_for_edit_and_ai].astype(str).tolist()
                        updated_master_df = st.session_state.data_for_coding_tab3.copy()
                        updated_master_df[id_col_for_edit_and_ai] = updated_master_df[id_col_for_edit_and_ai].astype(str)
                        
                        update_count = 0
                        for index, row in updated_master_df.iterrows():
                            if row[id_col_for_edit_and_ai] in ids_to_update:
                                existing_codes = robust_comma_string_to_list_p02(row['Codes'])
                                if code_val not in existing_codes: 
                                    existing_codes.append(code_val)
                                    updated_master_df.loc[index, 'Codes'] = robust_list_to_comma_string_p02(existing_codes)
                                    update_count += 1
                        
                        if update_count > 0:
                            st.session_state.data_for_coding_tab3 = updated_master_df
                            ui_helpers.show_success_message(f"Code '{code_val}' added to {update_count} displayed row(s).")
                            st.session_state[manual_code_input_key] = "" 
                            st.rerun()
                        else:
                            ui_helpers.show_info_message(f"Code '{code_val}' was already present in all displayed rows or no rows were updated.")
            if disable_manual_add and not df_for_coding_master_p02.empty:
                st.warning("Manual code addition is disabled because a unique ID column ('unique_app_id' or 'id') is missing from the data.")