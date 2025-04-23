from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from pathlib import Path
import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv
import io

# Load environment variables
load_dotenv()

app1 = FastAPI(title="BytemMe - ACE Alternative")

# Initialize OpenAI client
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting text from PDF: {str(e)}")

def process_with_chatgpt(text):
    """Process text with ChatGPT."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts and summarizes key information from documents."},
                {"role": "user", "content": f"Please analyze the following text and extract the key information:\n\n{text}"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing with ChatGPT: {str(e)}")

@app1.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
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
        
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(file_path)
        
        # Process with ChatGPT
        processed_info = process_with_chatgpt(pdf_text)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "File uploaded and processed successfully",
                "filename": file.filename,
                "file_path": str(file_path),
                "extracted_text": pdf_text,
                # "processed_info": processed_info
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing file: {str(e)}"}
        )
    finally:
        file.file.close()

@app1.get("/")
async def root():
    return {"message": "Welcome to the ByteMe"}