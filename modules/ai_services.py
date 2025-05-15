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
    # Try to find JSON within markdown code fences first
    match_fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
    if match_fence:
        logger.debug("Found JSON content within markdown code fences.")
        try:
            # Validate if the content within fences is valid JSON
            json.loads(match_fence.group(1).strip())
            return match_fence.group(1).strip()
        except json.JSONDecodeError:
            logger.warning("Content within markdown fences was not valid JSON. Falling back.")
            pass # Fall through if content in fence is not valid JSON itself

    # If no valid JSON in fences, try to find the first and last brackets/braces for arrays/objects
    # Attempt to find a JSON array first
    first_bracket = response_text.find('[')
    last_bracket = response_text.rfind(']')
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        potential_json_array = response_text[first_bracket : last_bracket + 1]
        try:
            json.loads(potential_json_array)
            logger.debug("Extracted potential JSON array directly.")
            return potential_json_array
        except json.JSONDecodeError:
            pass # Not a valid array, try object

    # Attempt to find a JSON object
    first_brace = response_text.find('{')
    last_brace = response_text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        potential_json_object = response_text[first_brace : last_brace + 1]
        try:
            json.loads(potential_json_object)
            logger.debug("Extracted potential JSON object directly.")
            return potential_json_object
        except json.JSONDecodeError:
            pass # Not a valid object

    logger.warning(f"Could not confidently extract JSON from response. Raw response might be used or an error might occur. Snippet: {response_text[:200]}")
    # If no JSON structure is found, return the original text, assuming it might be plain text (e.g. for summaries)
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
    cleaned_json_string = None # Initialize
    try:
        if provider_name == "OpenAI":
            client = client_or_module
            selected_model = model_name or "gpt-3.5-turbo"
            completion = client.chat.completions.create(model=selected_model, messages=[{"role": "user", "content": final_prompt}], temperature=0.1)
            ai_response_text_raw = completion.choices[0].message.content.strip()
        elif provider_name == "Gemini":
            gemini_module = client_or_module
            selected_model_gemini = model_name or "gemini-2.0-flash" # Make sure this is a valid model
            model_instance_gemini = gemini_module.GenerativeModel(selected_model_gemini, generation_config=genai.types.GenerationConfig(temperature=0.1))
            response_gemini = model_instance_gemini.generate_content(final_prompt)
            ai_response_text_raw = response_gemini.text.strip()

        if not ai_response_text_raw: raise ValueError("AI returned an empty response.")
        logger.debug(f"AI Raw Response (Code Gen) Snippet: {ai_response_text_raw[:500]}...")
        cleaned_json_string = extract_json_from_ai_response(ai_response_text_raw) # This might return non-JSON if that's what AI gave
        if not cleaned_json_string: raise ValueError("Could not extract a valid JSON structure from AI response for code gen.")
        logger.debug(f"Cleaned JSON String (Code Gen) Snippet for parsing: {cleaned_json_string[:500]}...")

        parsed_results = json.loads(cleaned_json_string) # This line expects cleaned_json_string to be valid JSON

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
        error_msg_detail = f"Failed to decode AI JSON response for code gen: {e}. Cleaned: '{cleaned_json_string[:200] if cleaned_json_string is not None else 'N/A'}'. Raw: '{ai_response_text_raw[:200] if ai_response_text_raw else 'N/A'}'"
        logger.error(error_msg_detail)
        return [{"unique_app_id": str(row[id_column_name]), id_column_name: str(row[id_column_name]), "Codes": "", "error": error_msg_detail} for _, row in batch_data_df.iterrows()]
    except Exception as e:
        error_msg_detail = f"Error during AI batch processing for code gen: {e}"
        logger.error(error_msg_detail, exc_info=True)
        return [{"unique_app_id": str(row[id_column_name]), id_column_name: str(row[id_column_name]), "Codes": "", "error": error_msg_detail} for _, row in batch_data_df.iterrows()]

def generate_code_groups_with_ai(unique_codes_list_str, prompt_template, provider_name, model_name=None):
    client_or_module = get_ai_client(provider_name)
    if not client_or_module:
        return {"error": "AI client not initialized for code grouping."}

    # Inject existing groups JSON if placeholder present
    # This part is usually handled by the calling page (themes.py) before passing the prompt_template
    # but if it's still here, it will be processed.
    if "{existing_groups_json}" in prompt_template:
        existing_groups_data = []
        # Assuming created_code_groups and edited_codebook_df are in session_state
        # This logic should ideally be in the calling page to avoid direct session_state access here
        cb_meta = st.session_state.get("edited_codebook_df", pd.DataFrame())
        for grp, codes in st.session_state.get("created_code_groups", {}).items():
            ids = []
            if not cb_meta.empty:
                for code in codes:
                    match = cb_meta[cb_meta["Code Name"] == code]
                    if not match.empty:
                        ids.append(str(match.iloc[0].get("code_id", "")))
            existing_groups_data.append({"group_name": grp, "codes": codes, "code_ids": ids})
        prompt_template = prompt_template.replace(
            "{existing_groups_json}",
            json.dumps(existing_groups_data, indent=2)
        )

    if "{unique_codes_list_json}" not in prompt_template:
        logger.error("Prompt template for AI code grouping is missing '{unique_codes_list_json}' placeholder.")
        return {"error": "Invalid AI prompt template for code grouping."}

    final_prompt = prompt_template.replace("{unique_codes_list_json}", unique_codes_list_str)
    logger.debug(f"Group-gen final prompt (first 500 chars):\n{final_prompt[:500]}")
    logger.info(f"Sending unique codes list to {provider_name} with model {model_name or 'default'} for group suggestion.")

    ai_response_text_raw = None
    cleaned_json_string = None
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
        st.session_state["last_ai_grouping_json"] = cleaned_json_string # Store for inspection dialog

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
                # Ensure code_ids are present if expected by the prompt, or add an empty list
                item.setdefault("code_ids", [])
                validated_groups.append(item)
            else:
                logger.warning(f"Skipping malformed group object from AI: {item}")

        if not validated_groups and parsed_results: # If AI returned something but none were valid
            raise ValueError("AI response parsed but contained no validly structured group objects.")

        return validated_groups

    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode AI JSON response for group gen: {e}. Cleaned: '{cleaned_json_string[:200] if cleaned_json_string else 'N/A'}'. Raw: '{ai_response_text_raw[:200] if ai_response_text_raw else 'N/A'}'"
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
                if not client: return None # Client init failed
                mdl = model_name or "gpt-4o" # Default to gpt-4o
                resp = client.chat.completions.create(
                    model=mdl,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.3, # Keep some variability for generation
                    max_tokens=4096, # Max output tokens
                )
                return resp.choices[0].message.content.strip()
            elif provider_name.lower() == "gemini":
                genai_module = get_ai_client("Gemini")
                if not genai_module: return None # Client init failed
                mdl = model_name or "gemini-2.0-flash" # Ensure this is a valid model
                model_instance = genai_module.GenerativeModel(mdl, generation_config=genai.types.GenerationConfig(temperature=0.3, max_output_tokens=4096))
                res = model_instance.generate_content(user_prompt)
                return res.text.strip()
            else:
                logger.error(f"Unsupported provider in _call_llm: {provider_name}")
                return None
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            return None # Indicate failure

    # ---- batching setup ----------------------------------------------
    if not batch_size or batch_size <= 0:
        batch_size = len(data_df) # Process all as one batch if not specified

    combined_entries = []
    seen_names = set(existing_code_names) # Track names to avoid duplicates within this run

    cursor = 0
    while cursor < len(data_df):
        chunk_df = data_df.iloc[cursor: cursor + batch_size]
        cursor += batch_size

        # Build data JSON for this chunk
        # Ensure text_column_name and id_column_name are correctly used
        json_data_batch_list = []
        for _, row in chunk_df.iterrows():
            item = {
                id_column_name: str(row.get(id_column_name, "")), # Handle if ID col is missing
                text_column_name: str(row.get(text_column_name, "")) if pd.notna(row.get(text_column_name)) else ""
            }
            json_data_batch_list.append(item)
        json_data_batch = json.dumps(json_data_batch_list, ensure_ascii=False)


        # Start from user template
        chunk_prompt = prompt_template_codebook.replace("{TEXT_COLUMN_NAME}", text_column_name)\
                                              .replace("{ID_COLUMN_NAME}", id_column_name)\
                                              .replace("{JSON_DATA_BATCH}", json_data_batch)
        # Prepend growing draft for later chunks (if any codes were generated in previous chunks)
        # This is essential for the iterative refinement within a single call to generate_codebook_with_ai
        if combined_entries: # If we have codes from previous chunks of THIS run
            draft_json_for_current_run = json.dumps([
                {
                    "Code": ent["Code Name"], # Assuming structure of combined_entries
                    "Description": ent["Description"],
                    "Rationale": ent["Rationale"],
                    "Example_ids": ent["Example_ids"],
                }
                for ent in combined_entries
            ], ensure_ascii=False)

            # The prompt should have a section like "CURRENT CODEBOOK (Draft from this run)"
            # The original prompt_template_codebook (from user input) should already contain
            # a placeholder for the OVERALL existing codebook (e.g., {existing_codebook_json})
            # which is filled by the calling page (codebook.py).
            # Here, we are adding codes generated *within this current execution of generate_codebook_with_ai*
            # to the prompt for subsequent batches *of this same execution*.
            # This needs careful prompt engineering to distinguish from the initial codebook.
            # For simplicity, if the main prompt already has a way to show the "full current codebook",
            # this internal batching refinement might add codes to that structure before sending.
            # The provided prompt has a merge_context for existing codes already.
            # The challenge is how to tell the AI about codes generated *just now in batch 1* when processing *batch 2*.
            # The current prompt template structure might not explicitly support this iterative refinement within batches well.
            # The most straightforward way with the current prompt structure is that the `prompt_template_codebook`
            # passed by `codebook.py` already contains ALL known codes up to the point of calling this function.
            # So, this internal `draft_json_for_current_run` might not be directly injectable unless the prompt supports it.
            # However, `codebook.py`'s prompt for AI codebook already injects the *entire current draft*
            # into the prompt template before calling this service. So this internal loop might be redundant if
            # `codebook.py` calls this service once per "Generate/Update" click.
            # If `codebook.py` intends to send the whole dataset at once and this function batches it,
            # then this `combined_entries` logic is for within-run consistency.
            #
            # Re-evaluating: The prompt `prompt_template_codebook` from `codebook.py` is the *full* prompt,
            # including the {existing_codebook_json} placeholder which is filled by `codebook.py`.
            # This function `generate_codebook_with_ai` then further replaces {JSON_DATA_BATCH}.
            # So, the `combined_entries` here are codes from *previous chunks of the current AI call*.
            # To make the AI aware of these, the prompt should ideally have a way to list them.
            #
            # Simpler approach: If the prompt is about generating codes *from the current batch*
            # and merging with an *existing codebook*, the `existing_code_names` and `seen_names`
            # are sufficient to prevent duplicates from being added to `combined_entries`.
            # The AI is told about the existing codebook once (via the initial prompt).
            pass # Not prepending combined_entries to prompt for now, relying on overall prompt structure.


        # ---- call the LLM --------------------------------------------
        raw = _call_llm(chunk_prompt)
        if raw is None: # LLM call failed
            return {"error": f"LLM call failed for batch starting at index {cursor - batch_size}."}

        cleaned_json_string = extract_json_from_ai_response(raw)
        if not cleaned_json_string:
             return {"error": f"Could not extract JSON from LLM response for batch starting at {cursor - batch_size}. Raw: {raw[:200]}"}
        try:
            parsed = json.loads(cleaned_json_string)
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error for batch starting at {cursor - batch_size}: {e}. Cleaned: {cleaned_json_string[:200]}"}

        if not isinstance(parsed, list):
            # If AI returns a single object instead of a list of one object
            if isinstance(parsed, dict) and {"Code", "Description", "Rationale", "Example_ids"} <= parsed.keys():
                parsed = [parsed] # Wrap it in a list
            else:
                return {"error": f"LLM did not return a JSON array of code objects for batch {cursor - batch_size}. Got: {type(parsed)}"}


        # ---- validate & merge ----------------------------------------
        for obj in parsed:
            if not isinstance(obj, dict):
                logger.warning(f"Skipping non-dict item in AI codebook response: {obj}")
                continue
            # Ensure all required keys are present
            if not {"Code", "Description", "Rationale", "Example_ids"} <= obj.keys():
                logger.warning(f"Skipping code object with missing keys: {obj}")
                continue

            code_name_from_ai = str(obj["Code"]).strip()
            if not code_name_from_ai:
                logger.warning(f"Skipping code object with empty code name: {obj}")
                continue

            # Rename "Code" to "Code Name" to match internal structure
            obj["Code Name"] = code_name_from_ai
            obj.pop("Code")


            if obj["Code Name"] in seen_names: # Avoid adding duplicates from this run
                logger.info(f"Skipping duplicate code (already seen in this run or existing): {obj['Code Name']}")
                continue

            # Ensure Example_ids is a list of strings
            ex_ids = obj.get("Example_ids")
            if isinstance(ex_ids, str):
                obj["Example_ids"] = [s.strip() for s in ex_ids.split(',') if s.strip()]
            elif not isinstance(ex_ids, list):
                obj["Example_ids"] = [] # Default to empty list if not string or list
            else: # It's a list, ensure elements are strings
                obj["Example_ids"] = [str(s).strip() for s in ex_ids if str(s).strip()]


            # Add other default fields if missing, though prompt should ensure they exist
            obj.setdefault("Description", "")
            obj.setdefault("Rationale", "")

            combined_entries.append(obj)
            seen_names.add(obj["Code Name"])

    logger.info(f"AI codebook generation run produced {len(combined_entries)} new/updated entries.")
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
        batch_size (int, optional): If provided and the data in prompt is large, attempts to chunk.
    Returns:
        list: A list of dicts, e.g., [{"id_column_name": "...", "Codes": "..."}, ...], or an error dict.
    """
    # --- Chunking logic (moved to the beginning) ---
    # This attempts to split the *data part* of the prompt if it's too large.
    # The `filled_prompt` contains both codebook JSON and data JSON.
    # This logic is a bit complex because it needs to parse the data JSON out, chunk it,
    # and then rebuild the prompt for each chunk.
    if isinstance(batch_size, int) and batch_size > 0:
        try:
            # Heuristic to find the start of the data JSON array.
            # Assumes codebook JSON comes first, then data JSON.
            # This might be fragile if prompt structure changes significantly.
            # A more robust way would be to pass data_list separately and let this function inject it.
            data_json_str_match = re.search(r"INPUT DATA\s*```json\s*(\[[\s\S]*?\])\s*```", filled_prompt, re.IGNORECASE | re.DOTALL)
            if not data_json_str_match: # Try without explicit "INPUT DATA" or markdown
                data_json_str_match = re.search(r"(\[[\s\S]*?\{[\s\S]*?\}\s*.*?\])", filled_prompt) # General JSON array

            if data_json_str_match:
                data_json_str = data_json_str_match.group(1)
                data_list_for_chunking = json.loads(data_json_str)

                if len(data_list_for_chunking) > batch_size:
                    logger.info(f"Data list length ({len(data_list_for_chunking)}) exceeds batch size ({batch_size}). Chunking requests.")
                    all_chunk_results = []
                    num_chunks = math.ceil(len(data_list_for_chunking) / batch_size)

                    for i in range(num_chunks):
                        chunk_data_list = data_list_for_chunking[i * batch_size : (i + 1) * batch_size]
                        chunk_data_json_str = json.dumps(chunk_data_list, indent=2)

                        # Reconstruct the prompt for this chunk
                        # This replaces the *entire matched data_json_str* with the new chunk's JSON.
                        chunk_prompt = filled_prompt.replace(data_json_str, chunk_data_json_str, 1) # Replace only the first occurrence

                        logger.info(f"Processing chunk {i+1}/{num_chunks} with {len(chunk_data_list)} items.")
                        # Recursive call without batch_size to process the chunk
                        chunk_result = generate_codes_with_codebook(
                            provider_name,
                            chunk_prompt,
                            id_column_name,
                            model_name,
                            batch_size=None # Don't re-batch a chunk
                        )
                        if isinstance(chunk_result, list):
                            all_chunk_results.extend(chunk_result)
                        elif isinstance(chunk_result, dict) and 'error' in chunk_result:
                            logger.error(f"Error processing chunk {i+1}: {chunk_result['error']}")
                            # Propagate the error for the whole operation if one chunk fails badly,
                            # or collect errors per item if possible. For now, fail fast.
                            return {"error": f"Error in chunk {i+1}: {chunk_result['error']}"}
                        else:
                            logger.warning(f"Unexpected result from chunk {i+1}: {type(chunk_result)}. Skipping.")
                    return all_chunk_results
            else:
                logger.warning("Could not identify data JSON in prompt for chunking. Proceeding with full prompt.")
        except json.JSONDecodeError as e:
            logger.error(f"JSONDecodeError during chunking setup: {e}. Could not parse data from prompt. Proceeding with full prompt.")
        except Exception as e:
            logger.error(f"Generic error during chunking setup: {e}. Proceeding with full prompt.", exc_info=True)
    # --- End of Chunking Logic ---


    client_or_module = get_ai_client(provider_name)
    if not client_or_module:
        logger.error("AI client not initialized for applying codebook.")
        return {"error": "AI client not initialized for applying codebook."}

    logger.info(f"Sending data to {provider_name} (model: {model_name or 'default'}) for applying codebook using filled prompt.")
    ai_response_text_raw = None
    cleaned_json_string = None
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

        # Remove any lingering markdown fences or backticks (extract_json_from_ai_response should handle this, but as a fallback)
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
            if isinstance(item, dict) and id_column_name in item: # "Codes" might be missing if AI decides none apply
                item_to_add = {
                    id_column_name: item[id_column_name],
                    "Codes": item.get("Codes", ""), # Default to empty string if "Codes" is missing
                    "code_ids": item.get("code_ids", "") # Also handle code_ids
                }
                validated_results.append(item_to_add)
            else:
                logger.warning(f"Skipping malformed row object from AI when applying codebook: {item}")

        if not validated_results and parsed_results: # If parsed_results was not empty but validated_results is
            raise ValueError("AI response for applying codebook parsed but contained no validly structured row objects.")

        return validated_results

    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode AI JSON response for applying codebook: {e}. Cleaned: '{cleaned_json_string[:200] if cleaned_json_string is not None else 'N/A'}'. Raw: '{ai_response_text_raw[:200] if ai_response_text_raw else 'N/A'}'"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        if "context_length_exceeded" in str(e).lower() or "prompt is too long" in str(e).lower() or "400_error" in str(e).lower() or "model's context length" in str(e).lower():
             error_msg = f"The dataset or codebook might be too large for a single AI request. Consider reducing batch size or data. Detail: {e}"
             logger.error(error_msg, exc_info=False) # exc_info=False as it's a common operational error
        else:
            error_msg = f"Error during AI application of codebook: {e}"
            logger.error(error_msg, exc_info=True)
        return {"error": error_msg}


def generate_summary_for_group_with_ai(group_name, text_samples_json, prompt_template, provider_name, model_name=None):
    """
    Generates a textual summary for a given thematic group based on text samples.

    Args:
        group_name (str): The name of the group being summarized.
        text_samples_json (str): A JSON string array of text samples from the group.
        prompt_template (str): The prompt template, expecting {GROUP_NAME} and {TEXT_SAMPLES_JSON}.
        provider_name (str): "OpenAI" or "Gemini".
        model_name (str, optional): Specific AI model name.

    Returns:
        tuple: (summary_text, raw_ai_response_str)
               On error, summary_text may contain an error message, and raw_ai_response_str will be an error dict.
    """
    client_or_module = get_ai_client(provider_name)
    if not client_or_module:
        error_detail = {"error": "AI client not initialized for summary generation."}
        return f"Error: {error_detail['error']}", json.dumps(error_detail)

    if "{GROUP_NAME}" not in prompt_template or "{TEXT_SAMPLES_JSON}" not in prompt_template:
        error_detail = {"error": "Prompt template for summary generation is missing required placeholders."}
        logger.error(error_detail["error"])
        return f"Error: {error_detail['error']}", json.dumps(error_detail)

    final_prompt = prompt_template.replace("{GROUP_NAME}", group_name)\
                                  .replace("{TEXT_SAMPLES_JSON}", text_samples_json)

    logger.info(f"Sending text samples for group '{group_name}' to {provider_name} (model: {model_name or 'default'}) for summary generation.")
    logger.debug(f"Summary gen final prompt (first 500 chars):\n{final_prompt[:500]}")

    ai_response_text_raw = None
    try:
        if provider_name == "OpenAI":
            client = client_or_module
            selected_model = model_name or "gpt-3.5-turbo" # Default model for summaries
            completion = client.chat.completions.create(
                model=selected_model,
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.3 # Slightly higher temp for more natural summaries
            )
            ai_response_text_raw = completion.choices[0].message.content.strip()
            # OpenAI usually returns the summary directly as text as per standard prompts
            summary_text = ai_response_text_raw
            # For consistency in return, the raw response *is* the summary text here if no JSON is expected
            raw_response_to_store = json.dumps({"summary_text_from_openai": summary_text, "full_completion_object": completion.model_dump_json(indent=2)})


        elif provider_name == "Gemini":
            gemini_module = client_or_module
            selected_model_gemini = model_name or "gemini-2.0-flash" # Default model
            model_instance_gemini = gemini_module.GenerativeModel(
                selected_model_gemini,
                generation_config=genai.types.GenerationConfig(temperature=0.3)
            )
            response_gemini = model_instance_gemini.generate_content(final_prompt)
            ai_response_text_raw = response_gemini.text.strip()
            summary_text = ai_response_text_raw
            # Storing the raw text and a structured version if possible.
            # Since prompt asks for plain text, response_gemini.parts might be more useful if it exists and has structured data
            try:
                raw_response_to_store = json.dumps({"summary_text_from_gemini": summary_text, "response_object": str(response_gemini)}) # Basic string representation
            except Exception:
                raw_response_to_store = json.dumps({"summary_text_from_gemini": summary_text, "error": "Could not serialize full Gemini response object."})


        if not summary_text: # If AI returned empty string
            summary_text = "(AI returned an empty summary)"
            logger.warning(f"AI returned an empty summary for group {group_name}.")
            if not raw_response_to_store: # Ensure raw_response_to_store is set
                 raw_response_to_store = json.dumps({"warning": "AI returned empty summary", "original_raw_response": ai_response_text_raw})


        logger.debug(f"AI Raw Summary Response for '{group_name}': {summary_text[:200]}...")
        return summary_text, raw_response_to_store

    except Exception as e:
        error_msg = f"Error during AI summary generation for group '{group_name}': {e}"
        logger.error(error_msg, exc_info=True)
        error_detail = {"error": error_msg, "raw_attempted_response": ai_response_text_raw if ai_response_text_raw else "N/A"}
        return f"Error generating summary: {e}", json.dumps(error_detail)


# Deprecated or less used function, ensure it's not called if not needed.
def generate_cluster_summary(cluster_texts, provider_name, model_name=None):
    # This function seems to be an older version or for a different purpose.
    # The new function generate_summary_for_group_with_ai is more specific.
    # If this is still needed, it should be updated to match the new return patterns.
    logger.warning("Deprecated function 'generate_cluster_summary' was called. Consider using 'generate_summary_for_group_with_ai'.")
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
                selected_model_gemini = model_name or "gemini-2.0-flash" # Ensure this is a valid model
                model_instance_gemini = client.GenerativeModel(selected_model_gemini) # client is genai module here
                response = model_instance_gemini.generate_content(prompt)
                summary = response.text.strip()
            return summary
    except Exception as e:
        error_msg = f"Error generating cluster summary with {provider_name}: {e}"
        logger.error(error_msg, exc_info=True)
        return f"Error generating summary: {str(e)}"