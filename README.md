# Themely: A Qualitative Thematic Analysis Tool

This Streamlit application is designed to assist qualitative researchers in performing thematic analysis on Reddit data. It allows users to authenticate with Reddit, download posts based on specific queries, and then leverage generative AI (OpenAI or Gemini) to generate codes, identify patterns, and derive themes from the text data.


**Development note:** This application was predominantly created with the assistance of AI code‑generation tools such as Google AI Studio and ChatGPT, as part of an experiment to evaluate the practical potential of these emerging technologies in real‑world research workflows.

## Features

*   **Project-Based Workflow:** Organize your analysis into distinct projects.
*   **Reddit Data Retrieval:**
    *   Fetch posts from specified subreddits.
    *   Filter by search queries, sort order, and timeframes.
*   **Interactive Data Management:**
    *   View and search fetched data in an interactive table.
    *   Edit data directly within the application (changes saved to "Processed Data").
    *   **PII Redaction:** Automatically redact sensitive information (like names, emails, phone numbers) using Microsoft Presidio Analyzer, with a new redacted column created.
*   **AI-Assisted Thematic Coding:**
    *   Choose between OpenAI or Google Gemini for generative AI tasks.
    *   **Codebook Generation:** Automatically create a shared codebook (concise set of tags) that captures the main patterns in your dataset.
    *   **Row‑wise Tagging:** Apply the generated codebook to label every Reddit post (up to 5 tags per item).
    *   View and manage AI-generated codes.
*   **Filtering and Views:**
    *   Filter data based on assigned codes.
    *   Save and load filtered "views" of your data for focused analysis.
*   **Clustering and Theme Generation:**
    *   Group related tags into higher‑level themes using embeddings and/or AI-driven clustering.
    *   Generate AI-driven summaries for (simulated) clusters to help identify overarching themes.
*   **Visualizations:**
    *   Generate word clouds from text data or AI codes.
    *   Display frequency charts for AI-generated codes.
*   **Secure API Key Management:** API keys are stored within project-specific configuration files, locally.

## Workflow Overview

1.  **Setup & Configuration (Tab 1):**
    *   **Create/Load Project:**
        *   Provide a project name and path.
    *   **API Key Management:**
        *   **Reddit API Credentials:** Obtain and enter your Reddit Client ID, Client Secret, and a unique User Agent.
        *   **Generative AI Keys:** Select your AI provider (OpenAI or Gemini) and enter the corresponding API key.
        *   Save keys to the project configuration.

2.  **Data Retrieval & Preparation (Tab 2):**
    *   **Fetch Reddit Data:** Specify subreddit, search query, number of posts, sort order, and time filter.
    *   **View/Edit Data:** Inspect raw and processed data in interactive tables. Make edits as needed (primarily on "Processed Data").
    *   **Redact Data:** Select a text column in your "Processed Data" for PII redaction using Microsoft Presidio. A new `_redacted` column will be created.

3.  **Codebook Generation, Tagging & View Management (Tab 3):**
    *   **AI‑Assisted Codebook & Tagging:**
        *   Select the text column (preferably a redacted one) for analysis.
        *   Generate a shared codebook of concise tags (saved as `codebook.json`).
        *   Apply that codebook to tag each row; the resulting tags are stored in a `tags` column.
        *   Review and, if necessary, edit tags directly in the table.
        *   Filter your "Processed Data" based on tags to create focused subsets.
    *   **Filter & Group:** Filter your "Processed Data" based on the generated `ai_codes`.
    *   **Save/Load Views:** Save filtered datasets as named "views" and load them later for focused analysis.

4.  **Analysis & Visualization (Tab 4):**
    *   **Clustering:** Cluster posts or tags to surface coherent groups automatically.
    *   **AI Theme Summaries:** Produce concise AI‑generated descriptions for each cluster to articulate overarching themes.
    *   **Visualizations:** Create word clouds (from text or codes) and bar charts of code frequencies.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Access to a Reddit account.
*   API keys for Reddit.
*   API key for either OpenAI or Google Gemini.

### Installation

1.  **Clone the repository (or download the files):**
    ```bash
    # If you have git installed
    # git clone <repository_url>
    # cd <repository_name>
    ```
    If you downloaded a ZIP, extract it.

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows
    # .venv\Scripts\activate
    # On macOS/Linux
    # source .venv/bin/activate
    ```

3.  **Install dependencies:**
    Navigate to the project's root directory (where `requirements.txt` is located) and run:
    ```bash
    pip install -r requirements.txt

### Running the Application

1.  Navigate to the project's root directory in your terminal.
2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
3.  The application will open in your web browser.

## Setting up API Credentials

### 1. Reddit API Credentials

To allow the application to fetch data from Reddit, you need to create a "script" type application on Reddit:

1.  **Log in to Reddit:**
    *   Go to [www.reddit.com](https://www.reddit.com) and log in.

2.  **Navigate to the "apps" page:**
    *   Go to: [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)

3.  **Click "are you a developer? create an app..." button:**
    *   Scroll down and click this button.

4.  **Fill out the application form:**
    *   **name:** A unique name for your app (e.g., `ThemelyApp_YourUsername`).
    *   **Type:** Select the **`script`** radio button.
    *   **description:** (Optional) E.g., "Streamlit app for thematic analysis."
    *   **about url:** (Optional) E.g., `http://localhost` or your GitHub repo URL.
    *   **redirect uri:** For script apps, a common placeholder is `http://localhost:8080`.

5.  **Click "create app".**

6.  **Get Your Credentials:**
    *   On the next page, you will find:
        *   **Client ID:** A string of characters **underneath your app's name** (labeled "personal use script").
        *   **Client Secret:** A string of characters next to the word **`secret`**.
    *   Copy these two values.

7.  **User Agent:**
    *   Reddit requires a unique User-Agent string. A good format is: `<platform>:<app_ID>:<version_string> by /u/<your_Reddit_username>`.
    *   Example: `Python:ThemelyApp:v0.1 by /u/MyRedditUserName123`

8.  **Enter in the App:**
    *   In Tab 1 of the Streamlit application, under "Reddit API Credentials", enter your Client ID, Client Secret, and the User Agent you created. Save the keys.

### 2. Generative AI API Keys (OpenAI or Gemini)

*   **OpenAI:**
    *   Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
    *   Create a new secret key. Copy it.
    *   In Tab 1 of the Streamlit app, select "OpenAI", paste your key, and save.
*   **Google Gemini:**
    *   Go to [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey) (or search for "Google AI Studio API Key").
    *   Create an API key. Copy it.
    *   In Tab 1 of the Streamlit app, select "Gemini", paste your key, and save.


## Project Structure

*   `app.py`: Main entry point for the Streamlit application.
*   `config.yaml`: (Currently minimal use) General application configuration.
*   `requirements.txt`: Python dependencies.
*   `.gitignore`: Specifies intentionally untracked files for Git.
*   `README.md`: This file.
*   `modules/`: Contains the core logic:
    *   `auth.py`: Handles authentication for API services and project setup.
    *   `reddit_api.py`: Manages Reddit API requests and data retrieval.
    *   `ai_services.py`: Manages interactions with OpenAI and Google Gemini APIs.
    *   `data_manager.py`: Handles data storage (local CSVs), redaction (Presidio), and view management.
    *   `ui_helpers.py`: UI components (notifications, tables, etc.).
    *   `utils.py`: Utility functions (config management, logging, error handling).
*   `data/`: (Created by the app if it doesn't exist, **should be in .gitignore**)
    *   This folder, or a user-specified folder for local projects, will contain subfolders for each project, storing:
        *   `{project_name}_config.yaml`: Project-specific configuration including API keys.
        *   `reddit_data.csv`: Raw data fetched from Reddit.
        *   `processed_data.csv`: Data after redaction, with AI codes, etc.
        *   `views/`: Subfolder for saved filtered data views.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for bugs, feature requests, or improvements.

## License

(Consider adding a license, e.g., MIT License)
This project is open-source and available under the [MIT License](LICENSE.txt). (You would need to create a LICENSE.txt file).