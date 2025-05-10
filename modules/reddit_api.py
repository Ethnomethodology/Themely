# modules/reddit_api.py

import streamlit as st
import praw
import pandas as pd
from . import ui_helpers, utils
import time

logger = utils.setup_logger(__name__)

def get_reddit_instance():
    """Initializes and returns a PRAW Reddit instance using stored API keys."""
    if 'project_config' not in st.session_state or not st.session_state.project_config.get('reddit_api'):
        ui_helpers.show_error_message("Reddit API keys not found in project configuration.")
        return None
    
    keys = st.session_state.project_config['reddit_api']
    try:
        reddit = praw.Reddit(
            client_id=keys.get('client_id'),
            client_secret=keys.get('client_secret'),
            user_agent=keys.get('user_agent')
            # Optional: username=keys.get('username'), password=keys.get('password') for script type apps
        )
        # Test connection
        reddit.user.me() # Or some other read-only, low-impact call
        logger.info("PRAW Reddit instance created and authenticated successfully.")
        return reddit
    except Exception as e:
        ui_helpers.show_error_message(f"Failed to initialize Reddit API: {e}")
        logger.error(f"PRAW initialization error: {e}")
        return None

def fetch_reddit_data(subreddit_name, query, limit=100, search_type="posts", time_filter="all", sort="relevance"):
    """
    Fetches posts or comments from a subreddit based on a query.
    search_type: 'posts' or 'comments' (comments search is more complex via PRAW, often done by iterating posts)
    sort: 'relevance', 'hot', 'top', 'new', 'comments' (for subreddit.search)
    time_filter: 'all', 'year', 'month', 'week', 'day', 'hour' (for subreddit.search)
    """
    reddit = get_reddit_instance()
    if not reddit:
        return None

    data_list = []
    
    # Initialize cancel flag in session state if not present
    if 'cancel_fetch' not in st.session_state:
        st.session_state.cancel_fetch = False
    st.session_state.cancel_fetch = False # Reset before fetch

    try:
        subreddit = reddit.subreddit(subreddit_name)
        progress_bar = st.progress(0, text="Initializing data retrieval...")
        status_message = st.empty()

        if search_type == "posts":
            status_message.info(f"Fetching posts from r/{subreddit_name} matching '{query}'...")
            # Using search within a subreddit
            # PRAW's search might not behave exactly like Reddit's UI search for all cases.
            # For general keyword search in a subreddit:
            search_results = subreddit.search(query, sort=sort, time_filter=time_filter, limit=limit)
            
            # If you want to get posts and then filter by keyword in title/body, it's a different approach:
            # submissions = subreddit.hot(limit=limit) # or .new(), .top()
            # relevant_submissions = [s for s in submissions if query.lower() in s.title.lower() or query.lower() in s.selftext.lower()]

            count = 0
            for post in search_results:
                if st.session_state.cancel_fetch:
                    ui_helpers.show_warning_message("Data retrieval cancelled by user. Partial data retained.")
                    logger.info("Reddit data fetch cancelled by user.")
                    break
                
                data_list.append({
                    'id': post.id,
                    'type': 'post',
                    'title': post.title,
                    'score': post.score,
                    'author': str(post.author),
                    'created_utc': post.created_utc,
                    'url': post.url,
                    'num_comments': post.num_comments,
                    'text': post.selftext,
                    'subreddit': subreddit_name
                })
                count += 1
                progress_bar.progress(min(count / limit, 1.0), text=f"Fetched {count}/{limit} posts...")
                # time.sleep(0.1) # Be respectful to API rate limits if fetching a lot

            if not st.session_state.cancel_fetch:
                ui_helpers.show_success_message(f"Data retrieval completed successfully. Fetched {len(data_list)} posts.")
        
        # elif search_type == "comments":
        #     # Fetching comments matching a query directly across a whole subreddit is not a direct PRAW feature.
        #     # Typically, you'd fetch posts, then their comments, and filter.
        #     # This is a simplified example; real comment search needs careful thought on scope.
        #     status_message.info(f"Fetching posts from r/{subreddit_name} to scan comments for '{query}'...")
        #     # Fetch posts first, then iterate through comments
        #     # This can be very slow and API intensive.
        #     posts_for_comments = subreddit.search(query, sort=sort, time_filter=time_filter, limit=max(10, limit // 10)) # Fetch fewer posts to scan comments
        #     temp_post_count = 0
        #     total_comments_scanned = 0
        #     for post in posts_for_comments:
        #         if st.session_state.cancel_fetch:
        #             break
        #         post.comments.replace_more(limit=0) # Expand top-level comments
        #         for comment in post.comments.list():
        #             if st.session_state.cancel_fetch:
        #                 break
        #             total_comments_scanned += 1
        #             if query.lower() in comment.body.lower():
        #                 data_list.append({
        #                     'id': comment.id,
        #                     'type': 'comment',
        #                     'post_id': post.id,
        #                     'post_title': post.title,
        #                     'score': comment.score,
        #                     'author': str(comment.author),
        #                     'created_utc': comment.created_utc,
        #                     'text': comment.body,
        #                     'parent_id': comment.parent_id,
        #                     'subreddit': subreddit_name
        #                 })
        #                 progress_bar.progress(min(len(data_list) / limit, 1.0) if limit > 0 else 0, 
        #                                     text=f"Scanned {total_comments_scanned} comments. Found {len(data_list)} matching comments...")
        #             if len(data_list) >= limit:
        #                 st.session_state.cancel_fetch = True # Force stop if limit reached
        #                 break
        #         if st.session_state.cancel_fetch and len(data_list) >= limit:
        #             break

        #     if not st.session_state.cancel_fetch:
        #         ui_helpers.show_success_message(f"Comment scan completed. Found {len(data_list)} comments.")
        #     elif st.session_state.cancel_fetch and len(data_list) < limit:
        #          ui_helpers.show_warning_message("Data retrieval cancelled by user. Partial data retained.")
        #     else: # Reached limit
        #         ui_helpers.show_success_message(f"Data retrieval completed. Reached limit of {limit} comments.")


        else:
            ui_helpers.show_error_message(f"Unsupported search type: {search_type}. Only 'posts' is currently well-supported.")
            return None

        progress_bar.empty() # Remove progress bar
        status_message.empty()
        return pd.DataFrame(data_list)

    except praw.exceptions.PRAWException as e:
        ui_helpers.show_error_message(f"Reddit API Error: {e}")
        logger.error(f"Reddit API error during fetch: {e}")
        if st.session_state.cancel_fetch: # If error happened after a cancel
             ui_helpers.show_warning_message("Data retrieval partially completed due to cancellation before error.")
        return pd.DataFrame(data_list) # Return any partial data
    except Exception as e:
        ui_helpers.show_error_message(f"Failed to retrieve data. Reason: {e}")
        logger.error(f"Generic error during Reddit data fetch: {e}")
        if st.session_state.cancel_fetch:
             ui_helpers.show_warning_message("Data retrieval partially completed due to cancellation before error.")
        return pd.DataFrame(data_list) # Return any partial data
    finally:
        st.session_state.cancel_fetch = False # Reset flag