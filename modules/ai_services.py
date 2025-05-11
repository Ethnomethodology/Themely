# modules/ai_services.py
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from . import ui_helpers, utils 
import json 
import pandas as pd 
import re # For extracting JSON

logger = utils.setup_logger(__name__)

def get_ai_client(provider_name):
    """Initializes and returns an AI client based on the provider."""
    if 'project_config' not in st.session_state:
        # ui_helpers.show_error_message("Project configuration not loaded.") # Can be too noisy
        logger.warning("AI Client: Project config not loaded.")
        return None

    api_key_info = st.session_state.project_config.get(f'{provider_name.lower()}_api', {})
    api_key = api_key_info.get('api_key')

    if not api_key:
        # ui_helpers.show_error_message(f"{provider_name} API key not found in project configuration.")
        logger.warning(f"AI Client: {provider_name} API key not found.")
        return None
    try:
        if provider_name == "OpenAI":
            client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized.")
            return client
        elif provider_name == "Gemini":
            genai.configure(api_key=api_key)
            logger.info("Gemini (Google Generative AI) client initialized.")
            return genai
        else:
            ui_helpers.show_error_message(f"Unsupported AI provider: {provider_name}")
            return None
    except Exception as e:
        ui_helpers.show_error_message(f"Failed to initialize {provider_name} client: {e}")
        return None


def extract_json_from_ai_response(response_text):
    """
    Attempts to extract a JSON array or object from a string that might contain
    markdown code fences (```json ... ```) or other text.
    """
    if not response_text:
        return None

    # Common pattern: ```json\n[...]\n``` or ```\n[...]\n```
    # Regex to find content within ```json ... ``` or just ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text, re.DOTALL)
    if match:
        logger.debug("Found JSON content within markdown code fences.")
        return match.group(1).strip() # Get the content inside the fences and strip whitespace

    # If no markdown fences, try to find the first '[' and last ']' or first '{' and last '}'
    # This is more brittle but can work if the JSON is just embedded.
    
    # Try to find a JSON array first
    first_bracket = response_text.find('[')
    last_bracket = response_text.rfind(']')
    if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
        potential_json_array = response_text[first_bracket : last_bracket + 1]
        try:
            json.loads(potential_json_array) # Test if it's valid JSON
            logger.debug("Extracted potential JSON array directly.")
            return potential_json_array
        except json.JSONDecodeError:
            pass # Not a valid JSON array

    # Try to find a JSON object if array fails
    first_brace = response_text.find('{')
    last_brace = response_text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        potential_json_object = response_text[first_brace : last_brace + 1]
        try:
            json.loads(potential_json_object) # Test if it's valid JSON
            logger.debug("Extracted potential JSON object directly.")
            return potential_json_object
        except json.JSONDecodeError:
            pass # Not a valid JSON object

    logger.warning(f"Could not confidently extract JSON from response. Returning raw response for parsing attempt. Snippet: {response_text[:200]}")
    return response_text # Fallback to returning the original text if no clear extraction


def generate_codes_for_batch_with_ai(batch_data_df, provider_name, prompt_template_batch, text_column_name, id_column_name="unique_app_id", model_name=None):
    client = get_ai_client(provider_name)
    default_error_return = [{"unique_app_id": row[id_column_name], "Codes": "", "error": "AI processing error."} for index, row in batch_data_df.iterrows()]

    if not client:
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
        return [{"unique_app_id": row[id_column_name], "Codes": "", "error": "Input text was empty."} for index, row in batch_data_df.iterrows()]

    json_data_batch_str = json.dumps(json_input_list, indent=2)

    if "{json_data_batch}" not in prompt_template_batch:
        logger.error("Prompt template for batch processing is missing '{json_data_batch}' placeholder.")
        return [{"unique_app_id": row[id_column_name], "Codes": "", "error": "Invalid AI prompt template for batch."} for index, row in batch_data_df.iterrows()]

    final_prompt = prompt_template_batch.replace("{json_data_batch}", json_data_batch_str)
    logger.info(f"Sending batch of {len(json_input_list)} items to {provider_name} with model {model_name or 'default'}.")
    
    ai_response_text_raw = None
    try:
        if provider_name == "OpenAI":
            selected_model = model_name or "gpt-3.5-turbo"
            completion = client.chat.completions.create(model=selected_model, messages=[{"role": "user", "content": final_prompt}], temperature=0.1)
            ai_response_text_raw = completion.choices[0].message.content.strip()
        elif provider_name == "Gemini":
            selected_model_gemini = model_name or "gemini-2.0-flash" # Ensure this model supports desired output
            model_instance_gemini = client.GenerativeModel(selected_model_gemini)
            # For Gemini, you might need to ensure the prompt explicitly requests JSON and the model supports it well.
            # Gemini 1.5 models can be configured for JSON output directly:
            # model_instance_gemini = client.GenerativeModel(
            #     selected_model_gemini,
            #     generation_config=genai.types.GenerationConfig(response_mime_type="application/json", temperature=0.1)
            # )
            response_gemini = model_instance_gemini.generate_content(final_prompt)
            ai_response_text_raw = response_gemini.text.strip()
        
        if not ai_response_text_raw:
            raise ValueError("AI returned an empty response.")

        logger.debug(f"AI Raw Response Snippet: {ai_response_text_raw[:500]}...")
        
        # --- Extract JSON from potentially dirty response ---
        cleaned_json_string = extract_json_from_ai_response(ai_response_text_raw)
        if not cleaned_json_string:
            raise ValueError("Could not extract a valid JSON structure from AI response.")
        
        logger.debug(f"Cleaned JSON String Snippet for parsing: {cleaned_json_string[:500]}...")
        parsed_results = json.loads(cleaned_json_string) # Attempt to parse the cleaned string

        if not isinstance(parsed_results, list):
            # Sometimes AI might return a single JSON object if the batch had only one item
            # or if it misunderstood. We expect a list.
            if isinstance(parsed_results, dict) and "unique_app_id" in parsed_results and "Codes" in parsed_results:
                logger.warning("AI returned a single JSON object, wrapping it in a list.")
                parsed_results = [parsed_results]
            else:
                raise ValueError("AI response, after cleaning, was not a JSON array or a parsable single JSON object.")

        ai_results_map = {str(item.get(id_column_name)): item.get("Codes", "") for item in parsed_results if id_column_name in item}
        
        final_batch_output = []
        for index, original_row in batch_data_df.iterrows():
            original_id_str = str(original_row[id_column_name])
            codes_from_ai = ai_results_map.get(original_id_str, "") 
            final_batch_output.append({
                "unique_app_id": original_id_str, # Use the original key from batch_data_df
                "Codes": codes_from_ai,
                "error": None
            })
        
        if len(final_batch_output) != len(batch_data_df):
            logger.warning(f"Mismatch in AI output count ({len(final_batch_output)}) vs batch input count ({len(batch_data_df)}). Some items might be missing codes or IDs didn't match.")
            # Fill any missing original items with an error
            original_ids_in_batch = {str(row[id_column_name]) for _, row in batch_data_df.iterrows()}
            returned_ids_in_output = {item['unique_app_id'] for item in final_batch_output}
            missing_ids = original_ids_in_batch - returned_ids_in_output
            for mid in missing_ids:
                final_batch_output.append({"unique_app_id": mid, "Codes": "", "error": "AI did not return data for this ID."})

        return final_batch_output

    except json.JSONDecodeError as e:
        error_msg = f"Failed to decode AI JSON response: {e}. Cleaned string snippet: {cleaned_json_string[:500] if 'cleaned_json_string' in locals() else 'N/A'}. Raw response snippet: {ai_response_text_raw[:500] if ai_response_text_raw else 'N/A'}"
        logger.error(error_msg)
        return [{"unique_app_id": str(row[id_column_name]), "Codes": "", "error": error_msg} for index, row in batch_data_df.iterrows()] # Return error for all items in batch
    except Exception as e:
        error_msg = f"Error during AI batch processing: {e}"
        logger.error(error_msg, exc_info=True)
        return [{"unique_app_id": str(row[id_column_name]), "Codes": "", "error": error_msg} for index, row in batch_data_df.iterrows()]


def generate_cluster_summary(cluster_texts, provider_name, model_name=None):
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
            elif provider_name == "Gemini":
                selected_model = model_name or "gemini-2.0-flash"
                model = client.GenerativeModel(selected_model)
                response = model.generate_content(prompt)
                summary = response.text.strip()
            return summary
    except Exception as e:
        error_msg = f"Error generating cluster summary with {provider_name}: {e}"
        ui_helpers.show_error_message(error_msg)
        return f"Error generating summary: {str(e)}"