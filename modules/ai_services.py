# modules/ai_services.py
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from . import ui_helpers, utils 
import json 
import pandas as pd 
import re 

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
            # Ensure genai is configured if not already done globally elsewhere
            # However, it's better to configure it each time to ensure the key is current.
            genai.configure(api_key=api_key) 
            logger.info("Gemini (Google Generative AI) client initialized (or re-initialized with key).")
            return genai # Return the genai module itself for Gemini
        else:
            # ui_helpers.show_error_message(f"Unsupported AI provider: {provider_name}") # Avoid UI call from module
            logger.error(f"Unsupported AI provider: {provider_name}")
            return None
    except Exception as e:
        # ui_helpers.show_error_message(f"Failed to initialize {provider_name} client: {e}")
        logger.error(f"Failed to initialize {provider_name} client: {e}", exc_info=True)
        return None


def extract_json_from_ai_response(response_text):
    """
    Attempts to extract a JSON array or object from a string that might contain
    markdown code fences (```json ... ```) or other text.
    """
    if not response_text:
        return None
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
        except json.JSONDecodeError:
            pass 
    first_brace = response_text.find('{')
    last_brace = response_text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        potential_json_object = response_text[first_brace : last_brace + 1]
        try:
            json.loads(potential_json_object) 
            logger.debug("Extracted potential JSON object directly.")
            return potential_json_object
        except json.JSONDecodeError:
            pass 
    logger.warning(f"Could not confidently extract JSON from response. Returning raw response for parsing attempt. Snippet: {response_text[:200]}")
    return response_text


def generate_codes_for_batch_with_ai(batch_data_df, provider_name, prompt_template_batch, text_column_name, id_column_name="unique_app_id", model_name=None):
    client_or_module = get_ai_client(provider_name) # For Gemini, this is the 'genai' module
    # For OpenAI, this is the OpenAI client instance

    default_error_return = [{"unique_app_id": row[id_column_name], "Codes": "", "error": "AI processing error."} for index, row in batch_data_df.iterrows()]

    if not client_or_module:
        return [{"unique_app_id": row[id_column_name], "Codes": "", "error": "AI client not initialized."} for index, row in batch_data_df.iterrows()]

    json_input_list = []
    for index, row in batch_data_df.iterrows():
        text_content = str(row[text_column_name]) if pd.notna(row[text_column_name]) else ""
        json_input_list.append({
            id_column_name: str(row[id_column_name]), 
            text_column_name: text_content 
        })

    if not json_input_list:
        logger.warning("Batch data for AI coding is empty after filtering.")
        # Return structure consistent with error for each item
        return [{"unique_app_id": str(row[id_column_name]), "Codes": "", "error": "Input text was empty."} for _, row in batch_data_df.iterrows()]


    json_data_batch_str = json.dumps(json_input_list, indent=2)

    if "{json_data_batch}" not in prompt_template_batch:
        logger.error("Prompt template for batch processing is missing '{json_data_batch}' placeholder.")
        return [{"unique_app_id": str(row[id_column_name]), "Codes": "", "error": "Invalid AI prompt template for batch."} for _, row in batch_data_df.iterrows()]

    final_prompt = prompt_template_batch.replace("{json_data_batch}", json_data_batch_str)
    logger.info(f"Sending batch of {len(json_input_list)} items to {provider_name} with model {model_name or 'default'} for code generation.")
    
    ai_response_text_raw = None
    try:
        if provider_name == "OpenAI":
            client = client_or_module # OpenAI client instance
            selected_model = model_name or "gpt-3.5-turbo" # Default if not specified
            completion = client.chat.completions.create(model=selected_model, messages=[{"role": "user", "content": final_prompt}], temperature=0.1)
            ai_response_text_raw = completion.choices[0].message.content.strip()
        elif provider_name == "Gemini":
            gemini_module = client_or_module # genai module
            selected_model_gemini = model_name or "gemini-1.5-flash-latest" 
            # For Gemini 1.5, explicitly ask for JSON output if possible, or ensure prompt is very clear.
            model_instance_gemini = gemini_module.GenerativeModel(
                selected_model_gemini,
                generation_config=genai.types.GenerationConfig(
                    # response_mime_type="application/json", # Enable if model supports direct JSON mode well
                    temperature=0.1 
                )
            )
            response_gemini = model_instance_gemini.generate_content(final_prompt)
            ai_response_text_raw = response_gemini.text.strip()
        
        if not ai_response_text_raw:
            raise ValueError("AI returned an empty response.")
        logger.debug(f"AI Raw Response (Code Gen) Snippet: {ai_response_text_raw[:500]}...")
        cleaned_json_string = extract_json_from_ai_response(ai_response_text_raw)
        if not cleaned_json_string:
            raise ValueError("Could not extract a valid JSON structure from AI response for code gen.")
        
        logger.debug(f"Cleaned JSON String (Code Gen) Snippet for parsing: {cleaned_json_string[:500]}...")
        parsed_results = json.loads(cleaned_json_string) 

        if not isinstance(parsed_results, list):
            if isinstance(parsed_results, dict) and id_column_name in parsed_results and "Codes" in parsed_results:
                logger.warning("AI returned a single JSON object for code gen, wrapping it in a list.")
                parsed_results = [parsed_results]
            else:
                raise ValueError("AI response for code gen, after cleaning, was not a JSON array or a parsable single JSON object.")

        # Map results by the ID column specified (could be 'unique_app_id' or other)
        ai_results_map = {str(item.get(id_column_name)): item.get("Codes", "") for item in parsed_results if id_column_name in item}
        
        final_batch_output = []
        for index, original_row in batch_data_df.iterrows():
            original_id_str = str(original_row[id_column_name])
            codes_from_ai = ai_results_map.get(original_id_str, "") 
            final_batch_output.append({
                "unique_app_id": original_id_str, # Ensure this key matches what the calling function expects
                id_column_name: original_id_str, # Also include the original ID column name if different
                "Codes": codes_from_ai,
                "error": None
            })
        
        if len(final_batch_output) != len(batch_data_df):
            logger.warning(f"Mismatch in AI output count ({len(final_batch_output)}) vs batch input count ({len(batch_data_df)}) for code gen.")
            original_ids_in_batch = {str(row[id_column_name]) for _, row in batch_data_df.iterrows()}
            returned_ids_in_output = {item[id_column_name] for item in final_batch_output}
            missing_ids = original_ids_in_batch - returned_ids_in_output
            for mid in missing_ids:
                error_item = {"unique_app_id": mid, "Codes": "", "error": "AI did not return data for this ID."}
                error_item[id_column_name] = mid # Ensure the dynamic id_column_name is also present
                final_batch_output.append(error_item)
        return final_batch_output

    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode AI JSON response for code gen: {e}. Cleaned: '{cleaned_json_string[:200] if 'cleaned_json_string' in locals() else 'N/A'}'. Raw: '{ai_response_text_raw[:200] if ai_response_text_raw else 'N/A'}'"
        logger.error(error_msg)
        return [{"unique_app_id": str(row[id_column_name]), id_column_name: str(row[id_column_name]), "Codes": "", "error": error_msg} for _, row in batch_data_df.iterrows()]
    except Exception as e:
        error_msg = f"Error during AI batch processing for code gen: {e}"
        logger.error(error_msg, exc_info=True)
        return [{"unique_app_id": str(row[id_column_name]), id_column_name: str(row[id_column_name]), "Codes": "", "error": error_msg} for _, row in batch_data_df.iterrows()]


def generate_code_groups_with_ai(unique_codes_list_str, prompt_template, provider_name, model_name=None):
    """
    Sends a list of unique codes (as a JSON string) to an AI to suggest thematic groups.
    Args:
        unique_codes_list_str (str): A JSON string representing a list of unique codes.
        prompt_template (str): The prompt template, expecting {unique_codes_list_json}.
        provider_name (str): "OpenAI" or "Gemini".
        model_name (str, optional): Specific model name.
    Returns:
        list: A list of dicts, e.g., [{"group_name": "...", "codes": [...]}, ...], or an error string/dict.
    """
    client_or_module = get_ai_client(provider_name)
    if not client_or_module:
        return {"error": "AI client not initialized for code grouping."}

    if "{unique_codes_list_json}" not in prompt_template:
        logger.error("Prompt template for AI code grouping is missing '{unique_codes_list_json}' placeholder.")
        return {"error": "Invalid AI prompt template for code grouping."}

    final_prompt = prompt_template.replace("{unique_codes_list_json}", unique_codes_list_str)
    logger.info(f"Sending unique codes list to {provider_name} with model {model_name or 'default'} for group suggestion.")
    
    ai_response_text_raw = None
    try:
        if provider_name == "OpenAI":
            client = client_or_module
            selected_model = model_name or "gpt-3.5-turbo" 
            completion = client.chat.completions.create(
                model=selected_model, 
                messages=[{"role": "user", "content": final_prompt}],
                temperature=0.2 # Slightly higher temp for creativity in grouping
            )
            ai_response_text_raw = completion.choices[0].message.content.strip()
        elif provider_name == "Gemini":
            gemini_module = client_or_module
            selected_model_gemini = model_name or "gemini-1.5-flash-latest"
            model_instance_gemini = gemini_module.GenerativeModel(
                selected_model_gemini,
                generation_config=genai.types.GenerationConfig(
                    # response_mime_type="application/json", # If model supports direct JSON reliably
                    temperature=0.2
                )
            )
            response_gemini = model_instance_gemini.generate_content(final_prompt)
            ai_response_text_raw = response_gemini.text.strip()
        
        if not ai_response_text_raw:
            raise ValueError("AI returned an empty response for code grouping.")

        logger.debug(f"AI Raw Response (Group Gen) Snippet: {ai_response_text_raw[:500]}...")
        cleaned_json_string = extract_json_from_ai_response(ai_response_text_raw)
        if not cleaned_json_string:
            raise ValueError("Could not extract a valid JSON structure from AI response for group gen.")
        
        logger.debug(f"Cleaned JSON String (Group Gen) Snippet for parsing: {cleaned_json_string[:500]}...")
        parsed_results = json.loads(cleaned_json_string)

        if not isinstance(parsed_results, list):
            # AI might return a single group object if it found only one
            if isinstance(parsed_results, dict) and "group_name" in parsed_results and "codes" in parsed_results:
                 logger.warning("AI returned a single group object, wrapping it in a list for group gen.")
                 parsed_results = [parsed_results]
            else:
                raise ValueError("AI response for group gen, after cleaning, was not a JSON array of groups or a single group object.")
        
        # Further validation of the structure of each item in parsed_results
        validated_groups = []
        for item in parsed_results:
            if isinstance(item, dict) and "group_name" in item and "codes" in item and isinstance(item["codes"], list):
                validated_groups.append(item)
            else:
                logger.warning(f"Skipping malformed group object from AI: {item}")
        
        if not validated_groups and parsed_results: # Parsed something, but not valid groups
             raise ValueError("AI response parsed but contained no validly structured group objects.")

        return validated_groups # List of {"group_name": ..., "codes": [...]}

    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode AI JSON response for group gen: {e}. Cleaned: '{cleaned_json_string[:200] if 'cleaned_json_string' in locals() else 'N/A'}'. Raw: '{ai_response_text_raw[:200] if ai_response_text_raw else 'N/A'}'"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Error during AI group generation: {e}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}


def generate_cluster_summary(cluster_texts, provider_name, model_name=None):
    # ... (existing function - no changes needed for this request) ...
    client = get_ai_client(provider_name)
    if not client: return "Error: AI client not initialized."
    combined_text = "\n\n---\n\n".join(cluster_texts)
    max_len = 30000 
    if len(combined_text) > max_len:
        combined_text = combined_text[:max_len] + "\n\n[Content truncated due to length]"
    prompt = f"The following are texts from a single thematic cluster. Provide a concise summary (2-3 sentences) that captures the core theme of this cluster:\n\n{combined_text}"
    try:
        with st.spinner(f"Generating summary with {provider_name}..."):
            if provider_name == "OpenAI":
                selected_model = model_name or "gpt-3.5-turbo"
                response = client.chat.completions.create(model=selected_model, messages=[{"role": "user", "content": prompt}])
                summary = response.choices[0].message.content.strip()
            elif provider_name == "Gemini": # Assumes client is the genai module
                selected_model = model_name or "gemini-1.5-flash-latest"
                model = client.GenerativeModel(selected_model)
                response = model.generate_content(prompt)
                summary = response.text.strip()
            return summary
    except Exception as e:
        error_msg = f"Error generating cluster summary with {provider_name}: {e}"
        # ui_helpers.show_error_message(error_msg) # Avoid direct UI from module
        logger.error(error_msg, exc_info=True)
        return f"Error generating summary: {str(e)}"