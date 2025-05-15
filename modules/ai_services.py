# modules/ai_services.py
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from . import ui_helpers, utils
import json
import pandas as pd
import re
import math # For sqrt in codebook generation

logger = utils.setup_logger(__name__)

def get_ai_client(provider_name):
    """Initializes and returns an AI client based on the provider."""
    if 'project_config' not in st.session_state:
        logger.warning("AI Client: Project config not loaded.")
        return None
    api_key_info = st.session_state.project_config.get(f'{provider_name.lower()}_api', {})
    api_key = api_key_info.get('api_key')
    if not api_key:
        logger.warning(f"AI Client: {provider_name} API key not found.")
        return None
    try:
        if provider_name == "OpenAI":
            client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized.")
            return client
        elif provider_name == "Gemini":
            genai.configure(api_key=api_key)
            logger.info("Gemini (Google Generative AI) client initialized (or re-initialized with key).")
            return genai
        else:
            logger.error(f"Unsupported AI provider: {provider_name}")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize {provider_name} client: {e}", exc_info=True)
        return None

def extract_json_from_ai_response(response_text):
    """
    Attempts to extract a JSON array or object from a string that might contain
    markdown code fences (```json ... ```) or other text.
    """
    if not response_text: return None
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
    if match:
        logger.debug("Found JSON content within markdown code fences.")
        return match.group(1).strip()
    first_bracket = response_text.find('[')
    last_bracket = response_text.rfind(']')
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        potential_json_array = response_text[first_bracket : last_bracket + 1]
        try:
            json.loads(potential_json_array)
            logger.debug("Extracted potential JSON array directly.")
            return potential_json_array
        except json.JSONDecodeError: pass
    first_brace = response_text.find('{')
    last_brace = response_text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        potential_json_object = response_text[first_brace : last_brace + 1]
        try:
            json.loads(potential_json_object)
            logger.debug("Extracted potential JSON object directly.")
            return potential_json_object
        except json.JSONDecodeError: pass
    logger.warning(f"Could not confidently extract JSON from response. Snippet: {response_text[:200]}")
    return response_text

def generate_codes_for_batch_with_ai(batch_data_df, provider_name, prompt_template_batch, text_column_name, id_column_name="unique_app_id", model_name=None):
    client_or_module = get_ai_client(provider_name)
    if not client_or_module:
        # Return a list of error dicts, one for each row in the batch_data_df
        return [{"unique_app_id": str(row[id_column_name]), id_column_name: str(row[id_column_name]), "Codes": "", "error": "AI client not initialized."} for _, row in batch_data_df.iterrows()]

    json_input_list = []
    for index, row in batch_data_df.iterrows():
        text_content = str(row[text_column_name]) if pd.notna(row[text_column_name]) else ""
        json_input_list.append({
            id_column_name: str(row[id_column_name]),
            text_column_name: text_content
        })
    if not json_input_list:
        logger.warning("Batch data for AI coding is empty after filtering.")
        return [{"unique_app_id": str(row[id_column_name]), id_column_name: str(row[id_column_name]), "Codes": "", "error": "Input text was empty."} for _, row in batch_data_df.iterrows()]

    json_data_batch_str = json.dumps(json_input_list, indent=2)
    if "{json_data_batch}" not in prompt_template_batch: 
        logger.error("Prompt template for batch processing is missing '{json_data_batch}' placeholder.")
        return [{"unique_app_id": str(row[id_column_name]), id_column_name: str(row[id_column_name]), "Codes": "", "error": "Invalid AI prompt template for batch."} for _, row in batch_data_df.iterrows()]

    final_prompt = prompt_template_batch.replace("{json_data_batch}", json_data_batch_str)
    # Note: {text_column_name} and {id_column_name} should have been replaced in prompt_template_batch *before* calling this function
    # if they are part of the main instruction text of the prompt, not specific to the json_data_batch structure.
    # The current prompt template structure implies they are part of the instructions around json_data_batch.

    logger.info(f"Sending batch of {len(json_input_list)} items to {provider_name} with model {model_name or 'default'} for code generation using template.")
    ai_response_text_raw = None
    try:
        if provider_name == "OpenAI":
            client = client_or_module
            selected_model = model_name or "gpt-3.5-turbo"
            completion = client.chat.completions.create(model=selected_model, messages=[{"role": "user", "content": final_prompt}], temperature=0.1)
            ai_response_text_raw = completion.choices[0].message.content.strip()
        elif provider_name == "Gemini":
            gemini_module = client_or_module
            selected_model_gemini = model_name or "gemini-2.0-flash"
            model_instance_gemini = gemini_module.GenerativeModel(selected_model_gemini, generation_config=genai.types.GenerationConfig(temperature=0.1))
            response_gemini = model_instance_gemini.generate_content(final_prompt)
            ai_response_text_raw = response_gemini.text.strip()
        
        if not ai_response_text_raw: raise ValueError("AI returned an empty response.")
        logger.debug(f"AI Raw Response (Code Gen) Snippet: {ai_response_text_raw[:500]}...")
        cleaned_json_string = extract_json_from_ai_response(ai_response_text_raw)
        if not cleaned_json_string: raise ValueError("Could not extract a valid JSON structure from AI response for code gen.")
        logger.debug(f"Cleaned JSON String (Code Gen) Snippet for parsing: {cleaned_json_string[:500]}...")
        
        parsed_results = json.loads(cleaned_json_string)
        
        if not isinstance(parsed_results, list):
            if isinstance(parsed_results, dict) and id_column_name in parsed_results and "Codes" in parsed_results:
                logger.warning("AI returned a single JSON object for code gen, wrapping it in a list.")
                parsed_results = [parsed_results]
            else: raise ValueError(f"AI response for code gen, after cleaning, was not a JSON array or a parsable single JSON object. Type: {type(parsed_results)}")
        
        ai_results_map = {str(item.get(id_column_name)): item.get("Codes", "") for item in parsed_results if id_column_name in item}
        final_batch_output = []
        
        for index, original_row in batch_data_df.iterrows():
            original_id_str = str(original_row[id_column_name])
            codes_from_ai = ai_results_map.get(original_id_str, "") # Default to empty string if ID not in AI response
            final_batch_output.append({"unique_app_id": original_id_str, id_column_name: original_id_str, "Codes": codes_from_ai, "error": None})
            
        if len(final_batch_output) != len(batch_data_df):
            logger.warning(f"Mismatch in AI output count ({len(final_batch_output)}) vs batch input count ({len(batch_data_df)}) for code gen.")
            # Ensure all original IDs get an entry, even if with an error/empty codes
            original_ids_in_batch = {str(row[id_column_name]) for _, row in batch_data_df.iterrows()}
            returned_ids_in_output = {item[id_column_name] for item in final_batch_output}
            missing_ids = original_ids_in_batch - returned_ids_in_output
            for mid in missing_ids:
                error_item = {"unique_app_id": mid, id_column_name: mid, "Codes": "", "error": "AI did not return data for this ID."}
                final_batch_output.append(error_item)
        return final_batch_output

    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode AI JSON response for code gen: {e}. Cleaned: '{cleaned_json_string[:200] if 'cleaned_json_string' in locals() and cleaned_json_string is not None else 'N/A'}'. Raw: '{ai_response_text_raw[:200] if ai_response_text_raw else 'N/A'}'"
        logger.error(error_msg)
        return [{"unique_app_id": str(row[id_column_name]), id_column_name: str(row[id_column_name]), "Codes": "", "error": error_msg} for _, row in batch_data_df.iterrows()]
    except Exception as e:
        error_msg = f"Error during AI batch processing for code gen: {e}"
        logger.error(error_msg, exc_info=True)
        return [{"unique_app_id": str(row[id_column_name]), id_column_name: str(row[id_column_name]), "Codes": "", "error": error_msg} for _, row in batch_data_df.iterrows()]

def generate_code_groups_with_ai(unique_codes_list_str, prompt_template, provider_name, model_name=None):
    client_or_module = get_ai_client(provider_name)
    if not client_or_module:
        return {"error": "AI client not initialized for code grouping."}
    # Inject existing groups JSON if placeholder present
    if "{existing_groups_json}" in prompt_template:
        existing = []
        cb_meta = st.session_state.get("edited_codebook_df", pd.DataFrame())
        for grp, codes in st.session_state.get("created_code_groups", {}).items():
            ids = []
            for code in codes:
                match = cb_meta[cb_meta["Code Name"] == code]
                if not match.empty:
                    ids.append(str(match.iloc[0]["code_id"]))
            existing.append({"group_name": grp, "codes": codes, "code_ids": ids})
        prompt_template = prompt_template.replace(
            "{existing_groups_json}",
            json.dumps(existing, indent=2)
        )
    if "{unique_codes_list_json}" not in prompt_template:
        logger.error("Prompt template for AI code grouping is missing '{unique_codes_list_json}' placeholder.")
        return {"error": "Invalid AI prompt template for code grouping."}
    final_prompt = prompt_template.replace("{unique_codes_list_json}", unique_codes_list_str)
    logger.debug(f"Group-gen final prompt:\n{final_prompt}")
    logger.info(f"Sending unique codes list to {provider_name} with model {model_name or 'default'} for group suggestion.")
    ai_response_text_raw = None
    try:
        if provider_name == "OpenAI":
            client = client_or_module
            selected_model = model_name or "gpt-3.5-turbo"
            completion = client.chat.completions.create(model=selected_model, messages=[{"role": "user", "content": final_prompt}], temperature=0.2)
            ai_response_text_raw = completion.choices[0].message.content.strip()
        elif provider_name == "Gemini":
            gemini_module = client_or_module
            selected_model_gemini = model_name or "gemini-2.0-flash"
            model_instance_gemini = gemini_module.GenerativeModel(selected_model_gemini, generation_config=genai.types.GenerationConfig(temperature=0.2))
            response_gemini = model_instance_gemini.generate_content(final_prompt)
            ai_response_text_raw = response_gemini.text.strip()
        if not ai_response_text_raw:
            raise ValueError("AI returned an empty response for code grouping.")
        logger.debug(f"AI Raw Response (Group Gen) Snippet: {ai_response_text_raw[:500]}...")
        cleaned_json_string = extract_json_from_ai_response(ai_response_text_raw)
        # Store raw grouping JSON for inspection
        st.session_state["last_ai_grouping_json"] = cleaned_json_string
        if not cleaned_json_string:
            raise ValueError("Could not extract a valid JSON structure from AI response for group gen.")
        logger.debug(f"Cleaned JSON String (Group Gen) Snippet for parsing: {cleaned_json_string[:500]}...")
        parsed_results = json.loads(cleaned_json_string)
        if not isinstance(parsed_results, list):
            if isinstance(parsed_results, dict) and "group_name" in parsed_results and "codes" in parsed_results:
                logger.warning("AI returned a single group object, wrapping it in a list for group gen.")
                parsed_results = [parsed_results]
            else:
                raise ValueError("AI response for group gen, after cleaning, was not a JSON array of groups or a single group object.")
        validated_groups = []
        for item in parsed_results:
            if isinstance(item, dict) and "group_name" in item and "codes" in item and isinstance(item["codes"], list):
                validated_groups.append(item)
            else:
                logger.warning(f"Skipping malformed group object from AI: {item}")
        if not validated_groups and parsed_results:
            raise ValueError("AI response parsed but contained no validly structured group objects.")
        return validated_groups
    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode AI JSON response for group gen: {e}. Cleaned: '{cleaned_json_string[:200] if 'cleaned_json_string' in locals() else 'N/A'}'. Raw: '{ai_response_text_raw[:200] if ai_response_text_raw else 'N/A'}'"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Error during AI group generation: {e}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}

def generate_codebook_with_ai(
    data_df: pd.DataFrame,
    provider_name: str,
    prompt_template_codebook: str,
    text_column_name: str,
    id_column_name: str,
    model_name: str = None,
    batch_size: int = None,
    existing_code_names: list = None
) -> "Union[List[Dict], Dict]":
    """
    Generates (or updates) a codebook using the chosen LLM.
    *Within one run* it processes the dataset in sequential batches and
    ensures every later batch sees the codes discovered in earlier batches.
    Parameters
    ----------
    data_df : pd.DataFrame
        Slice of the dataset to analyse.
    provider_name : {"OpenAI","Gemini"}
    prompt_template_codebook : str
        Prompt already contains CURRENT draft codebook JSON + merge rules and
        placeholders {TEXT_COLUMN_NAME}, {ID_COLUMN_NAME}, {JSON_DATA_BATCH}.
    text_column_name : str
    id_column_name   : str
    model_name       : str, optional
    batch_size       : int, optional
    existing_code_names : list[str], optional
        Used only for final exact‑string dedup at the end.
    Returns
    -------
    list[dict] – new codebook entries produced this run
    dict       – {"error": "..."} on failure
    """
    if existing_code_names is None:
        existing_code_names = []

    # ---- quick check prompt has placeholders -------------------------
    required_placeholders = ["{TEXT_COLUMN_NAME}", "{ID_COLUMN_NAME}", "{JSON_DATA_BATCH}"]
    if not all(ph in prompt_template_codebook for ph in required_placeholders):
        return {"error": "Prompt template missing required placeholders."}

    # ---- helper: single LLM call -------------------------------------
    def _call_llm(user_prompt: str):
        try:
            if provider_name.lower() == "openai":
                client = get_ai_client("OpenAI")
                mdl = model_name or "gpt-4o"
                resp = client.chat.completions.create(
                    model=mdl,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.3,
                    max_tokens=4096,
                )
                return resp.choices[0].message.content.strip()
            elif provider_name.lower() == "gemini":
                genai = get_ai_client("Gemini")
                mdl = model_name or "gemini-2.0-flash"
                res = genai.GenerativeModel(mdl).generate_content(user_prompt)
                return res.text.strip()
            else:
                return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return None

    # ---- batching setup ----------------------------------------------
    if not batch_size or batch_size <= 0:
        batch_size = len(data_df)

    combined_entries = []
    seen_names = set(existing_code_names)

    cursor = 0
    while cursor < len(data_df):
        chunk_df = data_df.iloc[cursor: cursor + batch_size]
        cursor += batch_size

        # Build data JSON for this chunk
        json_data_batch = json.dumps([
            {
                id_column_name: str(row[id_column_name]),
                text_column_name: str(row[text_column_name]) if pd.notna(row[text_column_name]) else ""
            }
            for _, row in chunk_df.iterrows()
        ], ensure_ascii=False)

        # Start from user template
        chunk_prompt = prompt_template_codebook.replace("{TEXT_COLUMN_NAME}", text_column_name)\
                                              .replace("{ID_COLUMN_NAME}", id_column_name)\
                                              .replace("{JSON_DATA_BATCH}", json_data_batch)

        # Prepend growing draft for later chunks
        if combined_entries:
            draft_json = json.dumps([
                {
                    "Code": ent["Code Name"],
                    "Description": ent["Description"],
                    "Rationale": ent["Rationale"],
                    "Example_ids": ent["Example_ids"],
                }
                for ent in combined_entries
            ], ensure_ascii=False)
            preamble = (
                "CURRENT CODEBOOK (Draft)\n"
                "Merge with these existing codes; do not duplicate.\n\n"
                f"{draft_json}\n\n"
            )
            chunk_prompt = preamble + chunk_prompt

        # ---- call the LLM --------------------------------------------
        raw = _call_llm(chunk_prompt)
        if raw is None:
            return {"error": "LLM call failed."}

        cleaned = extract_json_from_ai_response(raw)
        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            return {"error": f"JSON parse error: {e}"}

        if not isinstance(parsed, list):
            return {"error": "LLM did not return a JSON array."}

        # ---- validate & merge ----------------------------------------
        for obj in parsed:
            if not isinstance(obj, dict):
                continue
            if {"Code", "Description", "Rationale", "Example_ids"} <= obj.keys():
                obj["Code Name"] = obj.pop("Code")
                if obj["Code Name"] in seen_names:
                    continue
                if isinstance(obj["Example_ids"], str):
                    obj["Example_ids"] = [s.strip() for s in obj["Example_ids"].split(",") if s.strip()]
                elif not isinstance(obj["Example_ids"], list):
                    obj["Example_ids"] = []
                combined_entries.append(obj)
                seen_names.add(obj["Code Name"])

    return combined_entries

def generate_codes_with_codebook(provider_name, filled_prompt, id_column_name, model_name=None, batch_size=None):
    """
    Applies an existing codebook to a dataset using AI.
    The `filled_prompt` should already contain the codebook JSON and the data batch JSON.
    
    Args:
        provider_name (str): "OpenAI" or "Gemini".
        filled_prompt (str): The complete prompt string.
        id_column_name (str): Name of the column containing unique identifiers (used for parsing response).
        model_name (str, optional): Specific AI model name.
    Returns:
        list: A list of dicts, e.g., [{"id_column_name": "...", "Codes": "..."}, ...], or an error dict.
    """
    # Chunking support: split large prompts into smaller batches and combine results
    if isinstance(batch_size, int) and batch_size > 0:
        try:
            # Locate the data JSON array in the filled_prompt by finding the second '[{' (first is codebook, second is data batch)
            starts = [m.start() for m in re.finditer(r'\[\{', filled_prompt)]
            if len(starts) >= 2:
                data_start = starts[1]
            elif starts:
                data_start = starts[0]
            else:
                data_start = None
            if data_start is not None:
                # Find matching closing bracket for the JSON array
                depth = 0
                for i in range(data_start, len(filled_prompt)):
                    if filled_prompt[i] == '[':
                        depth += 1
                    elif filled_prompt[i] == ']':
                        depth -= 1
                        if depth == 0:
                            data_end = i
                            break
                data_json_str = filled_prompt[data_start:data_end+1]
                data_list = json.loads(data_json_str)
                # Only chunk if list length exceeds batch_size
                if len(data_list) > batch_size:
                    combined_results = []
                    for start_idx in range(0, len(data_list), batch_size):
                        chunk_list = data_list[start_idx:start_idx + batch_size]
                        chunk_json_str = json.dumps(chunk_list)
                        # Rebuild prompt with only this slice of data
                        chunk_prompt = filled_prompt[:data_start] + chunk_json_str + filled_prompt[data_end+1:]
                        # Recurse without batch_size to process each chunk
                        chunk_results = generate_codes_with_codebook(
                            provider_name,
                            chunk_prompt,
                            id_column_name,
                            model_name
                        )
                        # Merge chunk results, handling error dicts correctly
                        if isinstance(chunk_results, list):
                            combined_results.extend(chunk_results)
                        elif isinstance(chunk_results, dict):
                            combined_results.append(chunk_results)
                        else:
                            logger.warning(f"Unexpected chunk_results type in generate_codes_with_codebook: {type(chunk_results)}. Skipping.")
                    return combined_results
        except Exception as chunk_err:
            logger.warning(f"Chunking failed in generate_codes_with_codebook: {chunk_err}")
    client_or_module = get_ai_client(provider_name)
    if not client_or_module:
        logger.error("AI client not initialized for applying codebook.")
        return {"error": "AI client not initialized for applying codebook."}

    logger.info(f"Sending data to {provider_name} (model: {model_name or 'default'}) for applying codebook using filled prompt.")
    ai_response_text_raw = None
    try:
        if provider_name == "OpenAI":
            client = client_or_module
            selected_model = model_name or "gpt-3.5-turbo" 
            completion = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": filled_prompt}],
                temperature=0.1 
            )
            ai_response_text_raw = completion.choices[0].message.content.strip()
        elif provider_name == "Gemini":
            gemini_module = client_or_module
            selected_model_gemini = model_name or "gemini-2.0-flash"
            model_instance_gemini = gemini_module.GenerativeModel(
                selected_model_gemini,
                generation_config=genai.types.GenerationConfig(temperature=0.1)
            )
            response_gemini = model_instance_gemini.generate_content(filled_prompt) 
            ai_response_text_raw = response_gemini.text.strip()
        
        if not ai_response_text_raw:
            raise ValueError("AI returned an empty response when applying codebook.")

        logger.debug(f"AI Raw Response (Apply Codebook) Snippet: {ai_response_text_raw[:500]}...")
        cleaned_json_string = extract_json_from_ai_response(ai_response_text_raw)
        if not cleaned_json_string:
            raise ValueError("Could not extract a valid JSON array from AI response for applying codebook.")

        # Remove any lingering markdown fences or backticks
        cleaned_json_string = re.sub(r"^```(?:json)?\s*", "", cleaned_json_string).strip()
        cleaned_json_string = re.sub(r"\s*```$", "", cleaned_json_string).strip()

        logger.debug(f"Cleaned JSON String (Apply Codebook) Snippet for parsing: {cleaned_json_string[:500]}...")
        parsed_results = json.loads(cleaned_json_string)

        if not isinstance(parsed_results, list):
            if isinstance(parsed_results, dict) and id_column_name in parsed_results and "Codes" in parsed_results:
                 logger.warning("AI returned a single JSON object for applying codebook, wrapping it in a list.")
                 parsed_results = [parsed_results]
            else:
                raise ValueError(f"AI response for applying codebook, after cleaning, was not a JSON array of row objects. Type: {type(parsed_results)}")
        
        validated_results = []
        for item in parsed_results:
            if isinstance(item, dict) and id_column_name in item and "Codes" in item:
                item_to_add = {
                    id_column_name: item[id_column_name], 
                    "Codes": item["Codes"]
                }
                validated_results.append(item_to_add)
            else:
                logger.warning(f"Skipping malformed row object from AI when applying codebook: {item}")

        if not validated_results and parsed_results: # If parsed_results was not empty but validated_results is
            raise ValueError("AI response for applying codebook parsed but contained no validly structured row objects.")
            
        return validated_results

    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode AI JSON response for applying codebook: {e}. Cleaned: '{cleaned_json_string[:200] if 'cleaned_json_string' in locals() and cleaned_json_string is not None else 'N/A'}'. Raw: '{ai_response_text_raw[:200] if ai_response_text_raw else 'N/A'}'"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        if "context_length_exceeded" in str(e).lower() or "prompt is too long" in str(e).lower() or "400_error" in str(e).lower():
             error_msg = f"The dataset or codebook might be too large for a single AI request. Detail: {e}"
             logger.error(error_msg, exc_info=False) # exc_info=False as it's a common operational error
        else:
            error_msg = f"Error during AI application of codebook: {e}"
            logger.error(error_msg, exc_info=True)
        return {"error": error_msg}


def generate_cluster_summary(cluster_texts, provider_name, model_name=None):
    client = get_ai_client(provider_name)
    if not client: return "Error: AI client not initialized."
    combined_text = "\n\n---\n\n".join(cluster_texts)
    max_len = 30000 
    if len(combined_text) > max_len: combined_text = combined_text[:max_len] + "\n\n[Content truncated due to length]"
    prompt = f"The following are texts from a single thematic cluster. Provide a concise summary (2-3 sentences) that captures the core theme of this cluster:\n\n{combined_text}"
    try:
        with st.spinner(f"Generating summary with {provider_name}..."):
            if provider_name == "OpenAI":
                selected_model = model_name or "gpt-3.5-turbo"
                response = client.chat.completions.create(model=selected_model, messages=[{"role": "user", "content": prompt}])
                summary = response.choices[0].message.content.strip()
            elif provider_name == "Gemini":
                selected_model_gemini = model_name or "gemini-2.0-flash"
                model_instance_gemini = client.GenerativeModel(selected_model_gemini) # client is genai module here
                response = model_instance_gemini.generate_content(prompt)
                summary = response.text.strip()
            return summary
    except Exception as e:
        error_msg = f"Error generating cluster summary with {provider_name}: {e}"
        logger.error(error_msg, exc_info=True)
        return f"Error generating summary: {str(e)}"