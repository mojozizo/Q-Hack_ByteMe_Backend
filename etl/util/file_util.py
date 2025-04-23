from pathlib import Path

def create_or_get_upload_folder() -> Path:
    UPLOAD_DIR = Path("uploads")
    UPLOAD_DIR.mkdir(exist_ok=True)

    return UPLOAD_DIR