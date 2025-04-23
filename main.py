from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="BytemMe - ACE Alternative")

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Initialize OpenAI client
client = OpenAI()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def process_pdf_with_chatgpt(file_path: Path, query: str = None):
    """Process PDF directly with ChatGPT."""
    try:
        # Upload the file to OpenAI
        with open(file_path, "rb") as file:
            uploaded_file = client.files.create(
                file=file,
                purpose="assistants"
            )
        
        # Create an assistant
        assistant = client.beta.assistants.create(
            name="PDF Analyzer",
            instructions="You are a helpful assistant that analyzes PDF documents and answers questions about them.",
            model="gpt-4-turbo-preview",
            tools=[{"type": "retrieval"}]
        )
        
        # Create a thread
        thread = client.beta.threads.create()
        
        # Add the file to the thread
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=query or "Please analyze this PDF and provide a summary of its contents.",
            file_ids=[uploaded_file.id]
        )
        
        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        # Wait for the run to complete
        while True:
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status == "completed":
                break
        
        # Get the response
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        response = messages.data[0].content[0].text.value
        
        # Clean up
        client.files.delete(file_id=uploaded_file.id)
        client.beta.assistants.delete(assistant_id=assistant.id)
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing with ChatGPT: {str(e)}")

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), query: str = None):
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"message": "Only PDF files are allowed"}
            )
        
        # Save the file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process with ChatGPT
        processed_info = process_pdf_with_chatgpt(file_path, query)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded and processed successfully",
                "filename": file.filename,
                "file_path": str(file_path),
                "processed_info": processed_info
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing file: {str(e)}"}
        )
    finally:
        file.file.close()

@app.get("/")
async def root():
    return {"message": "Welcome to ByteMe"} 