import os


def get_newsapi_token() -> str:
    token = os.getenv("NEWSAPI_API_TOKEN")
    if not token:
        raise ValueError("NEWSAPI_API_TOKEN environment variable is not set.")
    return token

def get_brightdata_token() -> str:
    token = os.getenv("BRIGHTDATA_API_TOKEN")
    if not token:
        raise ValueError("BRIGHTDATA_API_TOKEN environment variable is not set.")
    return token

def get_brightdata_dataset_id() -> str:
    dataset_id = os.getenv("BRIGHTDATA_DATASET_ID")
    if not dataset_id:
        raise ValueError("BRIGHTDATA_DATASET_ID environment variable is not set.")
    return dataset_id