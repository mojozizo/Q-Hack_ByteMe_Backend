import os
import shutil
import json
import re
import time
from pathlib import Path as PathLib
from typing import Optional, Type, Dict, Any, Union
import PyPDF2

from fastapi import Path
from openai import OpenAI
from pydantic import BaseModel
from etl.extract.abstract_extracter import AbstractExtracter
from etl.util.file_util import create_or_get_upload_folder
from etl.util.web_search_util import WebSearchUtils
from etl.util.model_util import discover_nested_models, generate_extraction_prompt, generate_assistant_instructions, enrich_model_from_web, enrich_category_to_search
from models.model import Category, CompanyInfo, CategoryToSearch
from etl.agent import PDFAgentExecutor

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PDFExtracter(AbstractExtracter):
    """Extracts structured data from PDF pitch decks using OpenAI."""

    def __init__(self, model_class: Type[BaseModel] = Category, default_prompt: Optional[str] = None, 
                 enable_web_enrichment: bool = True, assistant_model: str = "gpt-4o",
                 use_agent_workflow: bool = False):
        """
        Initialize the PDFExtracter with customizable parameters.
        
        Args:
            model_class: The Pydantic model class to use for data structure (default: Category)
            default_prompt: A custom default prompt to use for extraction (default: built-in prompt)
            enable_web_enrichment: Whether to enrich data from web sources (default: True)
            assistant_model: The OpenAI model to use for analysis (default: gpt-4o)
            use_agent_workflow: Whether to use the LangChain agent workflow (default: False)
        """
        super().__init__()
        self.model_class = model_class
        self.default_prompt = default_prompt
        self.enable_web_enrichment = enable_web_enrichment
        self.assistant_model = assistant_model
        self.use_agent_workflow = use_agent_workflow
        # Use the utility function to discover nested models
        self.nested_fields = discover_nested_models(model_class)
        
        # Initialize agent executor if using agent workflow
        if self.use_agent_workflow:
            self.agent_executor = PDFAgentExecutor(model_name=assistant_model)

    def extract(self, file: Path, query: str = None) -> str:
        """
        Extracts structured information from a PDF file based on the configured model.
        If information is missing from the PDF and web enrichment is enabled, searches the web to fill in gaps.
        Also enriches with additional CategoryToSearch metrics from web sources.
        
        Args:
            file: The uploaded PDF file
            query: Custom query for analysis (optional)
            
        Returns:
            str: Structured JSON response containing extracted data and additional search metrics
        """
        # Save the uploaded file
        file_path = create_or_get_upload_folder() / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            # Choose extraction method based on configuration
            if self.use_agent_workflow:
                result = self._extract_with_agent(file_path, query)
            else:
                # Use existing implementation
                result = self._extract_with_openai_assistant(file_path, query)
                
            return json.dumps(result, indent=2)
            
        finally:
            file.file.close()
    
    def _extract_with_agent(self, file_path: PathLib, query: str = None) -> Dict[str, Any]:
        """
        Extracts data from a PDF using the LangChain agent workflow.
        
        Args:
            file_path: Path to the PDF file
            query: Custom query for analysis (optional)
            
        Returns:
            Dict: Structured data containing main_category and search_category
        """
        try:
            # Extract text from PDF
            pdf_text = self._extract_text_from_pdf(file_path)
            
            # Use the agent executor to extract data
            result = self.agent_executor.extract_from_pdf_text(
                pdf_text=pdf_text, 
                enable_web_enrichment=self.enable_web_enrichment
            )
            
            return result
            
        except Exception as e:
            print(f"Error extracting with agent: {str(e)}")
            # Return a valid but empty result if extraction fails
            return {
                "main_category": self.model_class().model_dump(),
                "search_category": CategoryToSearch().model_dump()
            }
    
    def _extract_text_from_pdf(self, file_path: PathLib) -> str:
        """
        Extracts text content from a PDF file using LangChain's document loaders.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            from langchain_community.document_loaders import PyPDFLoader
            
            # Use LangChain's PyPDFLoader
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Combine all document pages into a single text
            return "\n\n".join([doc.page_content for doc in documents])
        except Exception as e:
            print(f"Error extracting text from PDF with LangChain: {str(e)}")
            
            # Fallback to PyPDF2 if LangChain loader fails
            text = ""
            try:
                with open(file_path, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text += page.extract_text() + "\n\n"
                return text
            except Exception as e2:
                print(f"Fallback PDF extraction also failed: {str(e2)}")
                return ""
    
    def _extract_with_openai_assistant(self, file_path: PathLib, query: str = None) -> Dict[str, Any]:
        """
        Extracts data using the original OpenAI Assistant-based implementation.
        
        Args:
            file_path: Path to the PDF file
            query: Custom query for analysis (optional)
            
        Returns:
            Dict: Structured data containing main_category and search_category
        """
        # First, process the PDF and get structured output
        pdf_data_json = self._analyze_pdf(file_path, query)
        
        # Check if we got a valid response
        if not pdf_data_json or not pdf_data_json.strip():
            # Return a valid but empty JSON if the extraction failed
            print("Warning: Empty response from PDF analysis")
            empty_data = self.model_class().model_dump()
            return {"main_category": empty_data, "search_category": CategoryToSearch().model_dump()}
        
        try:
            # Parse the JSON string into a Python dict
            pdf_data = json.loads(pdf_data_json)
            
            # Enrich with web data if enabled
            if self.enable_web_enrichment:
                enriched_data = self._enrich_with_web_data(pdf_data)
                
                # Also get additional CategoryToSearch data
                company_name = None
                if self.model_class == Category and "company_info" in enriched_data:
                    company_name = enriched_data.get("company_info", {}).get("company_name")
                
                search_category_data = {}
                if company_name:
                    search_category = enrich_category_to_search(company_name)
                    search_category_data = search_category.model_dump()
                
                # Combine both data sets in the final result
                return {
                    "main_category": enriched_data,
                    "search_category": search_category_data
                }
            else:
                # Create a basic result with just the PDF data
                return {
                    "main_category": pdf_data,
                    "search_category": CategoryToSearch().model_dump()
                }
        except json.JSONDecodeError as e:
            print(f"Error parsing PDF analysis result as JSON: {str(e)}")
            print(f"Raw response: {pdf_data_json}")
            
            # Return a valid but empty JSON if the parsing failed
            empty_data = self.model_class().model_dump()
            return {"main_category": empty_data, "search_category": CategoryToSearch().model_dump()}

    def _enrich_with_web_data(self, pdf_data: dict) -> dict:
        """
        Enriches PDF-extracted data with information from web sources.
        Looks for missing fields in the PDF data and attempts to fill them.
        
        Args:
            pdf_data: The data extracted from the PDF
            
        Returns:
            dict: Enriched data combining PDF and web sources
        """
        # If using the default Category model, use existing enrichment logic
        if self.model_class == Category:
            return self._enrich_category_data(pdf_data)
        
        # For custom models, use our generic utility function
        try:
            # Validate the data with the model
            model_instance = self.model_class(**pdf_data)
            
            # Use the utility function to enrich the model
            enriched_model = enrich_model_from_web(
                model_instance=model_instance,
                web_search_util=WebSearchUtils
            )
            
            return enriched_model.model_dump()
        except Exception as e:
            print(f"Error enriching custom model data: {str(e)}")
            return pdf_data
    
    def _enrich_category_data(self, pdf_data: dict) -> dict:
        # Keep original implementation for backward compatibility
        # Create a validated Category object from PDF data
        try:
            category = Category(**pdf_data)
        except Exception as e:
            print(f"Error validating PDF data: {str(e)}")
            category = Category()
        
        # Get company info or create empty object if missing
        if not category.company_info:
            category.company_info = CompanyInfo()
        company_name = category.company_info.company_name
        
        if company_name:
            print(f"Enriching data for company: {company_name}")
            
            # If we have a website but missing company info fields, extract social profiles
            if category.company_info.website_link and not category.company_info.linkedin_profile_ceo:
                try:
                    social_profiles = WebSearchUtils.extract_social_profiles(category.company_info.website_link)
                    
                    # Update CEO LinkedIn profile if found
                    if "ceo_linkedin" in social_profiles and social_profiles["ceo_linkedin"]:
                        category.company_info.linkedin_profile_ceo = social_profiles["ceo_linkedin"]
                except Exception as e:
                    print(f"Social profile extraction failed: {str(e)}")
            
            # If we're missing basic company info, search for it
            missing_company_info = (
                not category.company_info.year_of_founding or 
                not category.company_info.location_of_headquarters or
                not category.company_info.industry or
                not category.company_info.business_model
            )
            
            if missing_company_info:
                try:
                    company_data = WebSearchUtils.search_company_info(company_name)
                    
                    # Update missing company info fields
                    if "year_of_founding" in company_data and not category.company_info.year_of_founding:
                        category.company_info.year_of_founding = company_data["year_of_founding"]
                    
                    if "location_of_headquarters" in company_data and not category.company_info.location_of_headquarters:
                        category.company_info.location_of_headquarters = company_data["location_of_headquarters"]
                    
                    if "industry" in company_data and not category.company_info.industry:
                        category.company_info.industry = company_data["industry"]
                    
                    if "business_model" in company_data and not category.company_info.business_model:
                        category.company_info.business_model = company_data["business_model"]
                    
                    if "employees" in company_data and not category.company_info.employees:
                        category.company_info.employees = company_data["employees"]
                    
                    if "website_link" in company_data and not category.company_info.website_link:
                        category.company_info.website_link = company_data["website_link"]
                    
                    if "one_sentence_pitch" in company_data and not category.company_info.one_sentence_pitch:
                        category.company_info.one_sentence_pitch = company_data["one_sentence_pitch"]
                except Exception as e:
                    print(f"Company info search failed: {str(e)}")
            
            # Try to get LinkedIn data
            if category.company_info.linkedin_profile_ceo:
                try:
                    # Extract name from LinkedIn URL if possible
                    ceo_first_name = None
                    ceo_last_name = None
                    linkedin_url = category.company_info.linkedin_profile_ceo
                    
                    if "/" in linkedin_url:
                        profile_name = linkedin_url.split("/")[-1]
                        if profile_name:
                            name_parts = profile_name.split("-")
                            if len(name_parts) >= 2:
                                ceo_first_name = name_parts[0]
                                ceo_last_name = " ".join(name_parts[1:])
                    
                    # Search LinkedIn using our utility
                    linkedin_data = WebSearchUtils.search_linkedin(
                        first_name=ceo_first_name,
                        last_name=ceo_last_name,
                        company_name=company_name,
                        profile_url=linkedin_url
                    )
                    
                    # Process LinkedIn data to extract relevant information
                    # This would be company-specific based on LinkedIn data structure
                except Exception as e:
                    print(f"LinkedIn data processing failed: {str(e)}")
            
            # Try to get news data 
            try:
                news_data = WebSearchUtils.search_news(company_name)
                
                # Process news data if needed
                # You could extract recent funding news, announcements, etc.
            except Exception as e:
                print(f"News data processing failed: {str(e)}")
            
            # Get missing financial metrics
            missing_financials = (
                not category.annual_recurring_revenue or
                not category.monthly_recurring_revenue or
                not category.customer_acquisition_cost or
                not category.customer_lifetime_value or
                not category.gross_margin or
                not category.burn_rate or
                not category.runway
            )
            
            if missing_financials:
                try:
                    financial_data = WebSearchUtils.search_financial_data(company_name)
                    
                    # Map financial data to category model
                    financial_fields = [
                        "annual_recurring_revenue", "monthly_recurring_revenue",
                        "customer_acquisition_cost", "customer_lifetime_value",
                        "cltv_cac_ratio", "gross_margin", "revenue_growth_rate_yoy",
                        "revenue_growth_rate_mom", "monthly_active_users",
                        "sales_cycle_length", "burn_rate", "runway"
                    ]
                    
                    # Update any missing fields that were found
                    for field in financial_fields:
                        if field in financial_data and not getattr(category, field, None):
                            setattr(category, field, financial_data[field])
                except Exception as e:
                    print(f"Financial data search failed: {str(e)}")
        
        # Return the enriched data as a dict
        return category.model_dump()

    def _analyze_pdf(self, pdf_path: PathLib, query: str = None) -> str:
        """Analyzes a PDF using OpenAI and returns structured data matching the model class."""
        # Use the provided query, default_prompt if set, or build a generic prompt
        if query:
            prompt = query
        elif self.default_prompt:
            prompt = self.default_prompt
        else:
            # Use the utility function to generate a prompt
            prompt = generate_extraction_prompt(self.model_class)
        
        # Use assistants API to handle PDF upload and analysis
        with open(pdf_path, "rb") as file:
            # Upload the file
            uploaded_file = client.files.create(
                file=file,
                purpose="assistants"
            )
            
            # Generate instructions using the utility function
            instructions = generate_assistant_instructions(self.model_class)
            
            # Create assistant to extract data based on the model
            assistant = client.beta.assistants.create(
                name="Document Analyzer",
                instructions=instructions,
                model=self.assistant_model,
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
                time.sleep(1)  # Added sleep to prevent tight polling
                
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
                
                # Parse and validate with the model
                parsed = json.loads(json_str)
                validated_data = self.model_class(**parsed)
                
                # Return the validated data as JSON
                return json.dumps(validated_data.model_dump(), indent=2)
            except Exception as e:
                # Return the raw response if parsing fails
                print(f"Failed to parse response as JSON: {str(e)}")
                return response_text
                
    def _build_extraction_prompt(self) -> str:
        """
        Creates a focused prompt to extract data from the document based on the model.
        If using the Category model, returns the specialized prompt, otherwise generates
        a generic prompt based on the model's fields.
        """
        # If using the default Category model, use the existing specialized prompt
        if self.model_class == Category:
            return self._build_category_extraction_prompt()
        
        # For other models, use the utility function
        return generate_extraction_prompt(self.model_class)
                
    def _build_category_extraction_prompt(self) -> str:
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