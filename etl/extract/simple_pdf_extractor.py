import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import UploadFile, File
from openai import OpenAI
from etl.extract.abstract_extracter import AbstractExtracter
from etl.util.file_util import create_or_get_upload_folder
from models.model import Category, CompanyInfo

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SimplePDFExtractor(AbstractExtracter):
    """A simplified extractor that uses OpenAI to extract information from PDF files."""

    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the PDF extractor.
        
        Args:
            model_name: The OpenAI model to use
        """
        super().__init__()
        self.model_name = model_name

    def extract(self, file: UploadFile, query: str = None) -> str:
        """
        Extract structured information from a PDF file.
        
        Args:
            file: The uploaded PDF file
            query: Custom extraction prompt (optional)
            
        Returns:
            str: JSON string with extracted information formatted according to the Category model
        """
        # Save the uploaded file
        file_path = create_or_get_upload_folder() / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Extract text from PDF
            pdf_text = self._extract_text_from_pdf(file_path)
            
            # Process the text with OpenAI and convert to Category model
            raw_json = self._process_with_openai(pdf_text, query)
            
            # Transform the raw data into our Category model
            category_data = self._transform_to_category_model(raw_json)
            
            # Return the JSON string of the Category model
            return json.dumps(category_data.dict(exclude_none=True), indent=2)
            
        except Exception as e:
            import traceback
            print(f"Error in PDF extraction: {str(e)}")
            print(traceback.format_exc())
            return json.dumps({"error": str(e)})
        finally:
            file.file.close()
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        try:
            # Use PyPDF2 for simplicity
            import PyPDF2
            
            text = ""
            with open(file_path, "rb") as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def _process_with_openai(self, pdf_text: str, query: str = None) -> Dict[str, Any]:
        """
        Process the PDF text with OpenAI to extract structured information.
        
        Args:
            pdf_text: The extracted text from the PDF
            query: Custom extraction prompt (optional)
            
        Returns:
            Dict: Raw extracted data
        """
        # Define the default extraction prompt if not provided
        if not query:
            query = """
            You are a specialized financial document analyzer. Your task is to extract ALL available information from this startup pitch deck. Be thorough and detailed.
            
            Extract the following information:
            
            1. COMPANY INFORMATION
               - Company Name
               - Official Company Name (if different)
               - Year of Founding (as integer)
               - Location of Headquarters
               - Business Model (B2B, B2C, etc.)
               - Industry
               - Required Funding Amount (as integer in USD)
               - Number of Employees (range like "10-50")
               - Website Link
               - One Sentence Pitch
               - LinkedIn Profile of CEO
               - Summary of the pitch deck highlighting key aspects
            
            2. FINANCIAL METRICS
               - Annual Recurring Revenue (ARR) in USD (as integer)
               - Monthly Recurring Revenue (MRR) in USD (as integer)
               - Customer Acquisition Cost (CAC) in USD (as integer)
               - Customer Lifetime Value (CLTV) in USD (as integer)
               - CLTV/CAC Ratio (as integer)
               - Gross Margin percentage (as integer)
               - Revenue Growth Rate YoY (as integer percentage)
               - Revenue Growth Rate MoM (as integer percentage)
            
            3. OPERATIONAL METRICS
               - Sales Cycle Length (in days, as integer)
               - Monthly Active Users (as integer)
               - User Growth Rate YoY (as integer percentage)
               - User Growth Rate MoM (as integer percentage)
               - Conversion Rate from free to paid (as integer percentage)
            
            4. STRATEGIC METRICS
               - Pricing Strategy Maturity (scale 1-5)
               - Burn Rate in USD monthly (as integer)
               - Runway in months (as integer)
               - IP Protection (1 for yes, 0 for no)
            
            5. MARKET METRICS
               - Market Competitiveness (scale 1-5)
               - Market Timing (scale 1-5)
               - Cap Table Cleanliness (scale 1-5)
            
            6. FOUNDER METRICS
               - Founder Industry Experience (years or scale 1-5)
               - Founder Past Exits (as integer)
               - Founder Background at target companies/universities (scale 1-5)
            
            7. LOCATION DATA
               - Country of Headquarters (as string)
            
            Structure the response as a JSON object with these fields exactly matching the Category model:
            {
              "company_info": {
                "company_name": string,
                "official_company_name": string,
                "year_of_founding": integer,
                "location_of_headquarters": string,
                "business_model": string,
                "industry": string,
                "required_funding_amount": integer,
                "employees": string,
                "website_link": string,
                "one_sentence_pitch": string,
                "linkedin_profile_ceo": string,
                "pitch_deck_summary": string
              },
              "annual_recurring_revenue": integer,
              "monthly_recurring_revenue": integer,
              "customer_acquisition_cost": integer,
              "customer_lifetime_value": integer,
              "cltv_cac_ratio": integer,
              "gross_margin": integer,
              "revenue_growth_rate_yoy": integer,
              "revenue_growth_rate_mom": integer,
              "sales_cycle_length": integer,
              "monthly_active_users": integer,
              "user_growth_rate_yoy": integer,
              "user_growth_rate_mom": integer,
              "conversion_rate": integer,
              "pricing_strategy_maturity": integer,
              "burn_rate": integer,
              "runway": integer,
              "ip_protection": integer,
              "market_competitiveness": integer,
              "market_timing": integer,
              "cap_table_cleanliness": integer,
              "founder_industry_experience": integer,
              "founder_past_exits": integer,
              "founder_background": integer,
              "country_of_headquarters": string
            }
            
            Be extremely thorough. Try to extract every possible value from the document. If you aren't certain about a value but have a reasonable estimate based on context, provide that value with your best judgment. Convert all percentages and monetary values to integers.
            
            Return only valid JSON matching this exact schema. Do not include null values.
            """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a specialized financial document analyzer that extracts structured information from startup pitch decks and financial documents. Be thorough and detailed in your extraction."},
                {"role": "user", "content": f"{query}\n\nDocument text:\n{pdf_text[:15000]}"}  # Limit text to prevent token overflow
            ],
            response_format={"type": "json_object"},
            temperature=0.1  # Lower temperature for more precise extraction
        )
        
        # Parse the JSON response
        result = json.loads(response.choices[0].message.content)
        return result
    
    def _transform_to_category_model(self, data: Dict[str, Any]) -> Category:
        """
        Transform raw extracted data into a Category model.
        
        Args:
            data: Raw data extracted from the PDF
            
        Returns:
            Category: Category model with transformed data
        """
        # Initialize an empty Category object
        category = Category()
        
        # Map company information if it exists
        if "company_info" in data and data["company_info"]:
            company_data = data["company_info"]
            company_info = CompanyInfo(
                company_name=company_data.get("company_name"),
                official_company_name=company_data.get("official_company_name"),
                year_of_founding=self._safe_int_convert(company_data.get("year_of_founding")),
                location_of_headquarters=company_data.get("location_of_headquarters"),
                business_model=company_data.get("business_model"),
                industry=company_data.get("industry"),
                required_funding_amount=self._safe_int_convert(company_data.get("required_funding_amount")),
                employees=company_data.get("employees"),
                website_link=company_data.get("website_link"),
                one_sentence_pitch=company_data.get("one_sentence_pitch"),
                linkedin_profile_ceo=company_data.get("linkedin_profile_ceo"),
                pitch_deck_summary=company_data.get("pitch_deck_summary")
            )
            category.company_info = company_info
        
        # Map all other category fields directly
        for field_name in category.model_fields:
            if field_name != "company_info" and field_name in data:
                # Check if field requires integer conversion
                if any(metric in field_name for metric in ["revenue", "rate", "cost", "value", "ratio", "users", "length", "burn", "runway", "maturity", "protection", "competitiveness", "timing", "cleanliness", "experience", "exits", "background"]):
                    setattr(category, field_name, self._safe_int_convert(data[field_name]))
                else:
                    setattr(category, field_name, data[field_name])
        
        # Extract country from location if available and not already set
        if not category.country_of_headquarters and category.company_info and category.company_info.location_of_headquarters:
            location = category.company_info.location_of_headquarters
            if "," in location:
                parts = location.split(",")
                country = parts[-1].strip()
                category.country_of_headquarters = country
            else:
                # Try to extract country code if present
                import re
                country_code_match = re.search(r'\b[A-Z]{2}\b', location)
                if country_code_match:
                    category.country_of_headquarters = country_code_match.group(0)
        
        return category
    
    def _safe_int_convert(self, value: Any) -> Optional[int]:
        """
        Safely convert a value to integer, handling different formats and returning None if not possible.
        
        Args:
            value: Value to convert to integer
            
        Returns:
            int or None: Converted integer or None if conversion not possible
        """
        if value is None:
            return None
            
        if isinstance(value, int):
            return value
            
        if isinstance(value, str):
            # Remove currency symbols, commas, and other non-numeric characters
            import re
            # First try to extract the number if there's text around it
            num_match = re.search(r'[\$€£¥]?[\s]*?(\d[\d,]*\.?\d*|\d+)', value)
            if num_match:
                cleaned = num_match.group(1).replace(',', '')
            else:
                cleaned = ''.join(c for c in value if c.isdigit() or c == '.')
            
            try:
                # Try to convert to float first, then to int
                return int(float(cleaned))
            except (ValueError, TypeError):
                # Try to handle "1K", "1M", etc.
                multipliers = {'k': 1000, 'm': 1000000, 'b': 1000000000}
                if len(value) > 1 and value[-1].lower() in multipliers:
                    try:
                        num_part = float(value[:-1])
                        return int(num_part * multipliers[value[-1].lower()])
                    except (ValueError, TypeError):
                        pass
                return None
                
        if isinstance(value, float):
            return int(value)
            
        return None