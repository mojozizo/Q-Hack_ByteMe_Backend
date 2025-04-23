from fastapi import UploadFile, File, APIRouter, HTTPException, Path
from starlette.responses import JSONResponse

from etl.extract.extractor_handler import ExtractorHandler
from etl.util.file_util import create_or_get_upload_folder

router = APIRouter()

@router.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), query: str = None):
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"message": "Only PDF files are allowed"}
            )

        file_path = create_or_get_upload_folder() / file.filename

        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded and processed successfully",
                "filename": file.filename,
                "file_path": str(file_path),
                "processed_info": ExtractorHandler.get_extractor("pdf").extract(file, query)
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing file: {str(e)}"}
        )
    finally:
        file.file.close()


@router.get("/")
async def root():
    return {"message": "Welcome to ByteMe"}