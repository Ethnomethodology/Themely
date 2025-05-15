# Themely: A Qualitative Thematic Analysis Tool

Themely is a Streamlit application designed to empower qualitative researchers in performing thematic analysis on Reddit data. It facilitates a project-based workflow, allowing users to authenticate with Reddit, retrieve posts based on specific queries, and then utilize generative AI (OpenAI or Google Gemini) to generate codes, identify patterns, derive themes, and create summaries from the textual data.

**Development Note:** This application is currently under active development. It was predominantly created with the assistance of AI code‑generation tools such as Google AI Studio and ChatGPT, as part of an experiment to evaluate the practical potential of these emerging technologies in real‑world research workflows. As such, it is provided "as is" without any warranties, and users should exercise their own discretion while using it for research purposes.


## Features

* **Project-Based Workflow**: Organize your research into distinct projects, each with its own configuration and data.
* **Multi-Page Interface**: Intuitive navigation through dedicated pages for each stage of the analysis: Project Setup, Data Management, Codebook, Themes, and Analysis.
* **Reddit Data Retrieval**:
    * Fetch posts from specified subreddits.
    * Filter by search queries, sort order (hot, new, top, relevance), and timeframes (all, year, month, week, day, hour).
* **Interactive Data Management**:
    * View and search fetched Reddit data.
    * Combine multiple downloaded datasets into "views" for analysis.
    * **PII Redaction**: Automatically redact sensitive information (like names, emails, phone numbers) from text columns using Microsoft Presidio Analyzer. Redacted data is typically used for AI processing.
* **AI-Assisted Thematic Coding & Codebook Management**:
    * Choose between OpenAI or Google Gemini for generative AI tasks.
    * **Codebook Generation**: Automatically create or update a project-specific codebook (`project_codebook.csv`) with AI, including code names, descriptions, rationales, and example post IDs.
    * Manually add, edit, or delete codebook entries.
    * Identify and merge semantically duplicate codes using local sentence embeddings.
    * **AI-Assisted Row-wise Tagging**: Apply the generated codebook to tag each data item (e.g., Reddit post) with relevant codes and their corresponding `code_ids`.
    * Manually update codes for individual data items.
* **Theme Generation**:
    * **AI-Assisted Theme Grouping**: Suggest thematic groups by clustering codes from the codebook using AI.
    * Manually create, edit, and manage groups of codes to form higher-level themes.
    * Assign codes to themes, ensuring a code belongs to only one theme.
* **Analysis & Summarization**:
    * Filter data based on assigned themes (groups).
    * **AI-Generated Summaries**: Generate concise AI-driven summaries for each thematic group based on its constituent data items.
* **Data Persistence**:
    * Projects, configurations, and data are stored locally.
    * Downloaded Reddit data, processed views, codebooks, and group summaries are saved in dedicated subfolders within each project directory.
* **Secure API Key Management**: API keys (Reddit, OpenAI/Gemini) are stored locally within project-specific configuration files (`<project_name_id>_config.yaml`).
* **Recent Projects Index**: Quickly access past projects via an SQLite database (`.themely_projects.db`) that stores project names and paths.

## Workflow Overview

The application guides you through a multi-page workflow:

1.  **Project Setup (`app.py` - Home Page)**
    * **Create or Open Project**:
        * Define a project name and specify a local directory for project files.
        * Open existing projects by selecting their configuration file or from a list of recent projects.
    * **API Key Management**:
        * Enter and save your Reddit API credentials (Client ID, Client Secret, User Agent).
        * Select your preferred Generative AI provider (OpenAI or Gemini) and enter the corresponding API key.
        * Keys are saved to the project's specific configuration file.

2.  **Data Management (`pages/data_management.py`)**
    * **Fetch Reddit Data**: Specify subreddit, search query, number of posts, sort order, and time filter to download data.
    * **Manage Downloaded Datasets**: View a list of previously downloaded raw data files.
    * **Combine & Create Views**: Select one or more downloaded datasets to combine them.
    * **Redact Data (Optional but Recommended)**: Select a text column in your combined data for PII redaction using Microsoft Presidio. Redaction is done in memory before view creation.
    * **Save View**: Save the combined (and optionally redacted) dataset as a named "project view" (`.csv` file and a `_meta.json` file) for subsequent analysis steps.

3.  **Codebook (`pages/codebook.py`)**
    * **Select View(s) for Coding**: Choose one or more saved project views to load data for codebook generation and application.
    * **AI-Assisted Codebook Generation**:
        * Select the text column (preferably redacted) for analysis.
        * Use AI (OpenAI/Gemini) to generate an initial codebook, suggesting codes, descriptions, rationales, and example IDs based on the selected data.
    * **Manual Codebook Management**:
        * Manually add new code entries.
        * Edit existing codes (name, description, rationale, example IDs).
        * Delete codes.
        * Find and merge duplicate codes.
    * **Inspect Code Examples**: View data items associated with example IDs for a code.
    * **Save Codebook**: Persist the draft codebook to `project_codebook.csv` within the project's `codes` subfolder.
    * **Apply Codes to Data**:
        * **AI-Assisted Coding**: Use AI to apply codes from your saved codebook to the selected data view(s). The results (codes and `code_ids`) are added as new columns to the data.
        * **Manual Coding**: Manually assign or update codes for selected rows in the data table.
        * Changes to coded data are saved back to the respective view file(s).

4.  **Themes (`pages/themes.py`)**
    * **Select Coded View(s)**: Choose views that have been processed with the codebook.
    * **AI-Assisted Theme Grouping**:
        * Use AI to suggest thematic groups by clustering codes from your codebook.
    * **Manual Theme Management**:
        * Create new theme groups.
        * Assign codes from the codebook to these groups. A code can only belong to one group.
        * Edit existing groups by adding or removing codes.
    * **Save Groups**: The grouping information updates the `groups` column in the data table and is saved back to the view files. A `groups_overview.csv` file is also saved in the project's `groups` subfolder.

5.  **Analysis (`pages/analysis.py`)**
    * **Select Grouped View(s)**: Choose views where themes (groups) have been defined.
    * **Explore Thematic Groups**:
        * View data items belonging to each specific theme/group.
        * Filter the displayed data table by a selected group.
    * **AI-Generated Group Summaries**:
        * For each theme/group, use AI to generate a concise textual summary based on a sample of its constituent text data.
        * Edit and save these summaries.
    * **Save Summaries**: Summaries are saved to `group_summaries.csv` in the project's `Analysis` subfolder.

## Getting Started

### Prerequisites

* Python 3.8+
* Access to a Reddit account.
* API keys for Reddit.
* API key for either OpenAI or Google Gemini.

### Installation

1.  **Clone the repository (or download the files):**
    ```bash
    # If you have git installed
    # git clone [https://github.com/Ethnomethodology/Themely.git](https://github.com/Ethnomethodology/Themely.git)
    # cd Themely
    ```
    If you downloaded a ZIP, extract it.

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv .venv
    # On Windows
    .venv\Scripts\activate
    # On macOS/Linux
    # source .venv/bin/activate
    ```

3.  **Install dependencies:**
    Navigate to the project's root directory (where `requirements.txt` is located) and run:
    ```bash
    pip install -r requirements.txt
    ```

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

1.  **Log in to Reddit**: Go to [www.reddit.com](https://www.reddit.com) and log in.
2.  **Navigate to the "apps" page**: Go to [https://www.reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)
3.  **Click "are you a developer? create an app..." button**.
4.  **Fill out the application form**:
    * **name**: A unique name for your app (e.g., `ThemelyApp_YourUsername`).
    * **Type**: Select the **`script`** radio button.
    * **description**: (Optional) E.g., "Streamlit app for thematic analysis."
    * **about url**: (Optional) E.g., `http://localhost` or your GitHub repo URL.
    * **redirect uri**: For script apps, a common placeholder is `http://localhost:8080`.
5.  **Click "create app"**.
6.  **Get Your Credentials**:
    * **Client ID**: A string of characters **underneath your app's name** (labeled "personal use script").
    * **Client Secret**: A string of characters next to the word **`secret`**.
    * Copy these two values.
7.  **User Agent**:
    * Reddit requires a unique User-Agent string. A good format is: `<platform>:<app_ID>:<version_string> by /u/<your_Reddit_username>`.
    * Example: `Python:ThemelyApp_YourProjectName:v0.1 by /u/MyRedditUserName123`
8.  **Enter in the App**:
    * On the "Project Setup" page of the Streamlit application, under "Reddit API", enter your Client ID, Client Secret, and the User Agent. Save the keys.

### 2. Generative AI API Keys (OpenAI or Gemini)

* **OpenAI**:
    * Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
    * Create a new secret key. Copy it.
    * On the "Project Setup" page, select "OpenAI", paste your key, and save.
* **Google Gemini**:
    * Go to [makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey) (or search for "Google AI Studio API Key").
    * Create an API key. Copy it.
    * On the "Project Setup" page, select "Gemini", paste your key, and save.

API keys are stored in a project-specific `<project_name_id>_config.yaml` file within the directory you choose for your project.

## Project Structure

* `app.py`: Main entry point for the Streamlit application; handles project setup and navigation.
* `pages/`: Contains modules for different pages/stages of the application:
    * `data_management.py`: Handles Reddit data download, view creation, and redaction.
    * `codebook.py`: Manages codebook creation (AI or manual), editing, and application to data.
    * `themes.py`: Facilitates grouping of codes into themes (AI or manual).
    * `analysis.py`: Enables analysis of themes and AI-powered summary generation for groups.
* `modules/`: Contains the core logic:
    * `auth.py`: Handles project setup (storage configuration) and API key storage.
    * `reddit_api.py`: Manages Reddit API requests and data retrieval using PRAW.
    * `ai_services.py`: Interfaces with OpenAI and Google Gemini APIs for various generative tasks (code generation, codebook creation, grouping, summaries).
    * `data_manager.py`: Handles local data storage (CSVs for downloads, views, codebooks, summaries), PII redaction (Presidio), and file/metadata listing.
    * `ui_helpers.py`: Provides helper functions for creating Streamlit UI components (notifications, tables, dialogs).
    * `utils.py`: Contains utility functions for configuration management (YAML), logging, and string sanitization.
    * `db.py`: Manages an SQLite database for indexing recent project paths.
* `config.yaml`: Minimal global application configuration (e.g., theme base).
* `requirements.txt`: Python dependencies for the project.
* `LICENSE.txt`: Project license information (MIT License).
* `README.md`: This file.
* **Project Data Folders (User-Defined Location)**: When a new project is created, a main folder for that project is established in the user-specified directory. Inside this project folder, the application creates subdirectories to store data:
    * `<project_name_id>_config.yaml`: Project-specific configuration, including API keys.
    * `reddit_downloads/`: Stores raw data CSV files fetched from Reddit (e.g., `reddit_data_subreddit_query_timestamp.csv`).
    * `project_views/`: Stores saved "views" as CSV files (e.g., `my_view.csv`) and corresponding metadata JSON files (e.g., `my_view_meta.json`). These views contain combined and potentially redacted/coded data.
    * `codes/`: Stores the project's codebook as `project_codebook.csv`.
    * `groups/`: Stores `groups_overview.csv` detailing code-to-group assignments.
    * `Analysis/`: Stores `group_summaries.csv` containing AI-generated or manually edited summaries for thematic groups.

## License

This project is open-source and available under the [MIT License](LICENSE.txt).