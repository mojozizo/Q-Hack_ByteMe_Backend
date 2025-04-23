from fastapi import UploadFile, File, APIRouter, HTTPException, Path, Query
from starlette.responses import JSONResponse

from etl.extract.extractor_handler import ExtractorHandler
from etl.util.file_util import create_or_get_upload_folder

router = APIRouter()

@router.post("/upload-pdf/")
async def upload_pdf(
    file: UploadFile = File(...), 
    query: str = None,
    use_agent_workflow: bool = Query(False, description="Whether to use LangChain agent workflow")
):
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"message": "Only PDF files are allowed"}
            )

        file_path = create_or_get_upload_folder() / file.filename
        
        try:
            # Process the file and get the extracted data
            extracted_data = ExtractorHandler.get_extractor(
                "pdf", 
                use_agent_workflow=use_agent_workflow
            ).extract(file, query)
            
            return JSONResponse(
                status_code=200,
                content={
                    "message": "File uploaded and processed successfully",
                    "filename": file.filename,
                    "file_path": str(file_path),
                    "processed_info": extracted_data,
                    "used_agent_workflow": use_agent_workflow
                }
            )
        except Exception as processing_error:
            # Log detailed error for debugging
            import traceback
            print(f"PDF processing error: {str(processing_error)}")
            print(traceback.format_exc())
            
            return JSONResponse(
                status_code=500,
                content={
                    "message": f"Error processing PDF content: {str(processing_error)}",
                    "filename": file.filename,
                    "file_path": str(file_path)
                }
            )
    except Exception as e:
        # Handle general exceptions
        import traceback
        print(f"General error: {str(e)}")
        print(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={"message": f"Server error: {str(e)}"}
        )
    finally:
        # Always close the file
        file.file.close()


@router.get("/")
async def root():
    return {"message": "Welcome to ByteMe"}