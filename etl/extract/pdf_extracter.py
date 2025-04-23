import os
import shutil
import json
import re
from pathlib import Path as PathLib

from fastapi import Path
from openai import OpenAI
from etl.extract.abstract_extracter import AbstractExtracter
from etl.util.file_util import create_or_get_upload_folder
from models.model import Category, CompanyInfo

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PDFExtracter(AbstractExtracter):
    """Extracts structured data from PDF pitch decks using OpenAI."""

    def __init__(self):
        super().__init__()

    def extract(self, file: Path, query: str = None) -> str:
        """
        Extracts structured information from a PDF pitch deck based on the Category model.
        
        Args:
            file: The uploaded PDF file
            query: Custom query for analysis (optional)
            
        Returns:
            str: Structured JSON response containing only Category model fields
        """
        # Save the uploaded file
        file_path = create_or_get_upload_folder() / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            # Process the PDF and get structured output
            return self._analyze_pdf(file_path, query)
        finally:
            file.file.close()

    def _analyze_pdf(self, pdf_path: PathLib, query: str = None) -> str:
        """Analyzes a PDF using OpenAI and returns structured data matching the Category model."""
        # Get the schema for structured output
        schema = Category.model_json_schema()
        company_info_schema = CompanyInfo.model_json_schema()
        
        # Use the default extraction prompt if none provided
        prompt = query if query else self._build_extraction_prompt()
        
        # Use assistants API to handle PDF upload and analysis
        with open(pdf_path, "rb") as file:
            # Upload the file
            uploaded_file = client.files.create(
                file=file,
                purpose="assistants"
            )
            
            # Create assistant to extract Category model fields
            assistant = client.beta.assistants.create(
                name="PDF Analyzer",
                instructions=f"""You are a specialized financial analyst for startups. 
                Analyze the pitch deck and extract ONLY the information specified in this schema:
                
                Category Schema:
                {json.dumps(schema, indent=2)}
                
                Company Info Schema (to be nested within Category):
                {json.dumps(company_info_schema, indent=2)}
                
                Important guidelines:
                1. All numeric values should be integers only (e.g., 15.5% becomes 16)
                2. For boolean values, use 1 for yes/true and 0 for no/false
                3. For scale metrics (like market_competitiveness), use values from 1-5
                4. Company information should be included in the nested "company_info" field
                5. Only include fields defined in the schema - do not add extra fields
                6. Return ONLY valid JSON matching the schema exactly
                7. Do not include any narrative analysis or additional text
                """,
                model="gpt-4o",
                tools=[{"type": "file_search"}]
            )
            
            # Create a thread with the query
            thread = client.beta.threads.create()
            
            # Add a message with the file attachment
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt,
                attachments=[
                    {
                        "file_id": uploaded_file.id,
                        "tools": [{"type": "file_search"}]
                    }
                ]
            )
            
            # Run the analysis
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            
            # Wait for completion
            while True:
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run_status.status == "completed":
                    break
                elif run_status.status in ["failed", "cancelled", "expired"]:
                    raise Exception(f"Analysis failed: {run_status.status}")
                
            # Get the response
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            # Extract the response content
            response_text = ""
            for message in messages.data:
                if message.role == "assistant":
                    for content in message.content:
                        if content.type == "text":
                            response_text += content.text.value
            
            # Clean up resources
            client.files.delete(uploaded_file.id)
            client.beta.assistants.delete(assistant_id=assistant.id)
            
            # Try to parse and validate as JSON
            try:
                # Extract just the JSON part (in case there's additional text)
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', response_text)
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                else:
                    json_str = response_text
                
                # Parse and validate with the Category model
                parsed = json.loads(json_str)
                validated_data = Category(**parsed)
                
                # Return only the Category model data as JSON
                return json.dumps(validated_data.model_dump(), indent=2)
            except Exception as e:
                # Return the raw response if parsing fails
                print(f"Failed to parse response as JSON: {str(e)}")
                return response_text
                
    def _build_extraction_prompt(self) -> str:
        """Creates a focused prompt to extract only Category model fields from the pitch deck."""
        return """Extract the following specific metrics from this pitch deck:

1. COMPANY INFORMATION
   - Company Name as a string
   - Official Company Name (if different) as a string
   - Year of Founding as an integer
   - Location of Headquarters as a string
   - Business Model as a string
   - Industry as a string
   - Required Funding Amount as an integer in USD
   - Number of Employees as a string (e.g., "10-50")
   - Website Link as a string
   - One Sentence Pitch as a string
   - LinkedIn Profile of CEO as a string
   - Detailed summary of the pitch deck highlighting all the important aspects of the company. 

2. FINANCIAL METRICS
   - Annual Recurring Revenue (ARR) in USD as an integer
   - Monthly Recurring Revenue (MRR) in USD as an integer
   - Customer Acquisition Cost (CAC) in USD as an integer
   - Customer Lifetime Value (CLTV) in USD as an integer
   - CLTV/CAC Ratio as an integer
   - Gross Margin percentage as an integer (e.g., 75% = 75)
   - Revenue Growth Rate year-over-year as an integer percentage
   - Revenue Growth Rate month-over-month as an integer percentage

3. OPERATIONAL METRICS
   - Sales Cycle Length in days as an integer
   - Monthly Active Users (MAU) as an integer
   - User Growth Rate year-over-year as an integer percentage
   - User Growth Rate month-over-month as an integer percentage
   - Conversion Rate from free to paid as an integer percentage

4. STRATEGIC METRICS
   - Pricing Strategy Maturity as an integer between 1-5
   - Burn Rate (monthly) in USD as an integer
   - Runway in months as an integer
   - IP Protection (1 for yes, 0 for no)

5. MARKET & COMPETITIVE METRICS
   - Market Competitiveness as an integer between 1-5
   - Market Timing advantage as an integer between 1-5
   - Cap Table Cleanliness as an integer between 1-5

6. FOUNDER & TEAM METRICS
   - Founder Industry Experience as an integer (years or scale 1-5)
   - Founder Past Exits as an integer
   - Founder Background/pedigree as an integer between 1-5

7. LOCATION DATA
   - Country of Headquarters as a string

Provide data ONLY for these specific fields and in the exact format requested. The Company Information should be nested under the "company_info" field. Return the data as a valid JSON object that follows the Category schema structure.
"""