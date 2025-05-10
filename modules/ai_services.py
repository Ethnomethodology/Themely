# modules/ai_services.py

import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from . import ui_helpers, utils

logger = utils.setup_logger(__name__)

def get_ai_client(provider_name):
    """Initializes and returns an AI client based on the provider."""
    if 'project_config' not in st.session_state:
        ui_helpers.show_error_message("Project configuration not loaded.")
        return None

    api_key_info = st.session_state.project_config.get(f'{provider_name.lower()}_api', {})
    api_key = api_key_info.get('api_key')

    if not api_key:
        ui_helpers.show_error_message(f"{provider_name} API key not found in project configuration.")
        return None

    try:
        if provider_name == "OpenAI":
            client = OpenAI(api_key=api_key)
            # Test call (optional, or rely on first actual use)
            # client.models.list() 
            logger.info("OpenAI client initialized.")
            return client
        elif provider_name == "Gemini":
            genai.configure(api_key=api_key)
            # Test call (optional)
            # for model in genai.list_models(): pass
            logger.info("Gemini (Google Generative AI) client initialized.")
            return genai # The genai module itself is used for requests
        else:
            ui_helpers.show_error_message(f"Unsupported AI provider: {provider_name}")
            return None
    except Exception as e:
        ui_helpers.show_error_message(f"Failed to initialize {provider_name} client: {e}")
        logger.error(f"Error initializing {provider_name} client: {e}")
        return None

def generate_codes_with_ai(text_data, provider_name, prompt_template, model_name=None):
    """
    Generates initial codes for a list of texts using the chosen AI provider.
    text_data: A list of strings or a single string.
    prompt_template: A string template for the prompt, e.g., "Generate 3-5 thematic codes for the following text: {text}"
    """
    client = get_ai_client(provider_name)
    if not client:
        return None

    results = []
    is_single_text = isinstance(text_data, str)
    if is_single_text:
        text_data = [text_data]

    progress_bar = st.progress(0, text=f"Preparing to generate codes with {provider_name}...")
    
    for i, text_item in enumerate(text_data):
        if not text_item or not isinstance(text_item, str) or len(text_item.strip()) < 10 : # Basic check for empty or very short text
            results.append({"text": text_item, "codes": [], "error": "Text too short or invalid."})
            progress_bar.progress((i + 1) / len(text_data), text=f"Skipping item {i+1}/{len(text_data)} (invalid text)...")
            continue

        prompt = prompt_template.format(text=text_item)
        try:
            progress_bar.progress((i + 1) / len(text_data), text=f"Generating codes for item {i+1}/{len(text_data)}...")
            if provider_name == "OpenAI":
                selected_model = model_name or "gpt-3.5-turbo" # Default model
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                generated_codes_text = response.choices[0].message.content.strip()
            elif provider_name == "Gemini":
                selected_model = model_name or "gemini-pro" # Default model
                model = client.GenerativeModel(selected_model)
                response = model.generate_content(prompt)
                generated_codes_text = response.text.strip()
            
            # Assuming codes are returned as a comma-separated list or newline-separated
            codes = [code.strip() for code in generated_codes_text.replace('\n', ',').split(',') if code.strip()]
            results.append({"text": text_item, "codes": codes, "error": None})
            logger.info(f"Successfully generated codes for text item {i+1} using {provider_name}.")

        except Exception as e:
            error_msg = f"Error generating codes with {provider_name} for item {i+1}: {e}"
            ui_helpers.show_error_message(error_msg) # Show error for this item
            logger.error(error_msg)
            results.append({"text": text_item, "codes": [], "error": str(e)})
    
    progress_bar.empty()
    ui_helpers.show_success_message(f"AI coding process completed for {len(text_data)} items.")
    return results if not is_single_text else results[0]


def generate_cluster_summary(cluster_texts, provider_name, model_name=None):
    """Generates an AI-driven summary for a cluster of texts."""
    client = get_ai_client(provider_name)
    if not client:
        return "Error: AI client not initialized."

    combined_text = "\n\n---\n\n".join(cluster_texts)
    # Truncate if too long to avoid exceeding token limits (simple truncation)
    # A more sophisticated approach would be to summarize chunks then summarize summaries.
    max_len = 30000 # Approximate character limit, adjust based on model
    if len(combined_text) > max_len:
        combined_text = combined_text[:max_len] + "\n\n[Content truncated due to length]"
        
    prompt = f"The following are texts from a single thematic cluster. Provide a concise summary (2-3 sentences) that captures the core theme of this cluster:\n\n{combined_text}"

    try:
        with st.spinner(f"Generating summary with {provider_name}..."):
            if provider_name == "OpenAI":
                selected_model = model_name or "gpt-3.5-turbo"
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[{"role": "user", "content": prompt}]
                )
                summary = response.choices[0].message.content.strip()
            elif provider_name == "Gemini":
                selected_model = model_name or "gemini-pro"
                model = client.GenerativeModel(selected_model)
                response = model.generate_content(prompt)
                summary = response.text.strip()
            logger.info(f"Successfully generated cluster summary using {provider_name}.")
            return summary
    except Exception as e:
        error_msg = f"Error generating cluster summary with {provider_name}: {e}"
        ui_helpers.show_error_message(error_msg)
        logger.error(error_msg)
        return f"Error generating summary: {str(e)}"

# Add functions for theme generation, etc., as needed following similar patterns.