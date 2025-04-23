from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

app = FastAPI(title="ByteMe - ACE Alternative")

# Get API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Warning: OPENAI_API_KEY environment variable is not set")
    print("Please create a .env file with your OpenAI API key:")
    print("OPENAI_API_KEY=your_api_key_here")
    # Don't raise an error, just set a flag
    api_key = "not_set"

# Initialize OpenAI client - completely simplified to avoid proxies error
# The OpenAI client will automatically use the OPENAI_API_KEY environment variable
client = OpenAI()

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def process_pdf_with_chatgpt(file_path: Path, query: str = None):
    """Process PDF directly with ChatGPT."""
    if api_key == "not_set":
        raise HTTPException(
            status_code=500,
            detail="OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable."
        )
    
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
            instructions="You are a helpful assistant that analyzes Startup Pitch decks and answers questions about them.",
            model="gpt-4-turbo-preview",
            tools=[{"type": "file_search"}]
        )
        
        # Create a thread
        thread = client.beta.threads.create()
        
        # Add the file to the thread
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=query or "Please analyze the pitch deck and provide a summary of the company from a venture capital point of view ",
            attachments=[{
                "file_id": uploaded_file.id,
                "tools": [{"type": "file_search"}]
            }]
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
            elif run_status.status in ["failed", "cancelled", "expired"]:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Run failed with status: {run_status.status}"
                )
            # Sleep to avoid excessive API calls
            time.sleep(1)
        
        # Get the response
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        # Check if there are messages and if they contain content
        if not messages.data:
            return "No response was generated."
        
        # Get the most recent assistant message
        assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
        if not assistant_messages or not assistant_messages[0].content:
            return "No response content was found."
        
        # Extract text from the message content
        response = ""
        for content_item in assistant_messages[0].content:
            if hasattr(content_item, 'type') and content_item.type == 'text':
                response = content_item.text.value
                break
        
        # Clean up
        client.files.delete(uploaded_file.id)
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