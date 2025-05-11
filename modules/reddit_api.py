# modules/reddit_api.py
import streamlit as st
import praw
import pandas as pd
from . import ui_helpers, utils 
import time
from datetime import datetime # For unique ID timestamp

logger = utils.setup_logger(__name__)

def get_reddit_instance():
    if 'project_config' not in st.session_state or not st.session_state.project_config.get('reddit_api'):
        logger.warning("Attempted to get Reddit instance without API keys in project config.")
        return None
    keys = st.session_state.project_config['reddit_api']
    if not keys.get('client_id') or not keys.get('client_secret') or not keys.get('user_agent'):
        logger.warning("Reddit API credentials incomplete.")
        return None
    try:
        reddit = praw.Reddit(
            client_id=keys.get('client_id'),
            client_secret=keys.get('client_secret'),
            user_agent=keys.get('user_agent')
        )
        for _ in reddit.subreddit("all").hot(limit=1): pass # Basic API call test
        logger.info("PRAW Reddit instance created and basic API call successful.")
        return reddit
    except praw.exceptions.PRAWException as e:
        ui_helpers.show_error_message(f"Failed to initialize Reddit API (PRAW Error): {e}")
        return None
    except Exception as e:
        ui_helpers.show_error_message(f"Failed to initialize Reddit API: {e}")
        return None

def fetch_reddit_data(subreddit_name, query, limit=100, search_type="posts", time_filter="all", sort="relevance"):
    reddit = get_reddit_instance()
    if not reddit: return None

    data_list = []
    if 'cancel_fetch' not in st.session_state: st.session_state.cancel_fetch = False
    st.session_state.cancel_fetch = False 

    # Timestamp for this specific fetch operation to help create unique IDs
    fetch_timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S%f") # Added microseconds for more uniqueness

    try:
        subreddit_instance = reddit.subreddit(subreddit_name)
        try: _ = subreddit_instance.display_name 
        except Exception as sub_e:
            ui_helpers.show_error_message(f"Subreddit 'r/{subreddit_name}' not found or access denied: {sub_e}")
            return None

        progress_bar_placeholder = st.empty()
        status_message_placeholder = st.empty()
        
        if query:
            status_message_placeholder.info(f"Searching r/{subreddit_name} for '{query}' (sort: {sort}, time: {time_filter})...")
            fetch_iterator = subreddit_instance.search(query, sort=sort, time_filter=time_filter, limit=limit)
        elif sort == "hot":
            status_message_placeholder.info(f"Fetching 'hot' posts from r/{subreddit_name}...")
            fetch_iterator = subreddit_instance.hot(limit=limit)
        elif sort == "new":
            status_message_placeholder.info(f"Fetching 'new' posts from r/{subreddit_name}...")
            fetch_iterator = subreddit_instance.new(limit=limit)
        elif sort == "top":
            status_message_placeholder.info(f"Fetching 'top' posts from r/{subreddit_name} (time: {time_filter})...")
            fetch_iterator = subreddit_instance.top(time_filter=time_filter, limit=limit)
        else: 
            status_message_placeholder.info(f"Fetching 'hot' posts from r/{subreddit_name} (defaulted)...")
            fetch_iterator = subreddit_instance.hot(limit=limit)

        count = 0
        progress_text_template = "Fetching post {current}/{total}..."
        if limit > 0: progress_bar_placeholder.progress(0, text=progress_text_template.format(current=0, total=limit))

        for i, post in enumerate(fetch_iterator): # Use enumerate for a serial number within this fetch
            if st.session_state.cancel_fetch:
                ui_helpers.show_warning_message("Data retrieval cancelled by user.")
                break
            
            # Create a unique ID for each item within the app
            # Combining fetch timestamp, subreddit, original reddit ID, and a serial number from this fetch
            unique_app_id = f"{fetch_timestamp_str}_{subreddit_name}_{post.id}_{i}"

            data_list.append({
                'unique_app_id': unique_app_id, # Newly added unique ID
                'id': post.id, # Original Reddit ID
                'type': 'post',
                'title': post.title,
                'score': post.score,
                'author': str(post.author), 
                'created_utc': post.created_utc,
                'url': post.url,
                'num_comments': post.num_comments,
                'text': post.selftext, 
                'subreddit': subreddit_name,
                'link_flair_text': post.link_flair_text
            })
            count += 1
            if limit > 0: progress_bar_placeholder.progress(min(count / limit, 1.0), text=progress_text_template.format(current=count, total=limit))
            # time.sleep(0.01) # Small delay if needed

        progress_bar_placeholder.empty(); status_message_placeholder.empty()
        if not st.session_state.cancel_fetch:
            if data_list: ui_helpers.show_success_message(f"Data retrieval completed. Fetched {len(data_list)} posts.")
            else: ui_helpers.show_warning_message(f"No posts found matching your criteria in r/{subreddit_name}.")
        
        return pd.DataFrame(data_list) if data_list else pd.DataFrame()
    except praw.exceptions.PRAWException as e:
        ui_helpers.show_error_message(f"Reddit API Error: {e}")
        return pd.DataFrame(data_list) 
    except Exception as e:
        ui_helpers.show_error_message(f"Failed to retrieve data from Reddit. Reason: {e}")
        logger.error(f"Generic error during Reddit data fetch: {e}", exc_info=True)
        return pd.DataFrame(data_list)
    finally:
        st.session_state.cancel_fetch = False