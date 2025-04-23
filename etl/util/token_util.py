import os


def get_newsapi_token() -> str:
    token = os.getenv("NEWSAPI_API_TOKEN")
    if not token:
        raise ValueError("NEWSAPI_API_TOKEN environment variable is not set.")
    return token