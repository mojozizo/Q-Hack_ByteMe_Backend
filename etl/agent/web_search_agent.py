from typing import Dict, Any, Optional, List
import json
import os

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate, SystemMessage
from pydantic import BaseModel

from etl.util.web_search_util import WebSearchUtils
from models.model import Category, CategoryToSearch


class WebSearchAgent:
    """
    Agent to enhance PDF extraction results using web search.
    This agent runs after the PDF agent to fill in missing information
    by searching the web for relevant data.
    """

    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the web search agent.
        
        Args:
            model_name: The name of the OpenAI model to use
        """
        self.model_name = model_name
        self.llm = ChatOpenAI(temperature=0.3, model=model_name)
    
    def enhance_results(self, pdf_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the results from the PDF agent with web search data.
        
        Args:
            pdf_results: Results from the PDF agent
            
        Returns:
            Enhanced results combining PDF and web data
        """
        # Extract the main category and company info
        main_category = pdf_results.get("main_category", {})
        search_category = pdf_results.get("search_category", {})
        
        # Extract company name
        company_name = self._extract_company_name(main_category)
        
        if not company_name:
            print("No company name found, cannot enhance results with web search")
            return pdf_results
        
        print(f"Enhancing results for company: {company_name}")
        
        # Enhance main category
        enhanced_main = self._enhance_main_category(main_category, company_name)
        
        # Get additional search category data
        enhanced_search = self._enhance_search_category(search_category, company_name)
        
        # Return the enhanced results
        return {
            "main_category": enhanced_main,
            "search_category": enhanced_search,
            "source": "pdf+web",
            "enhanced_by": "web_search_agent"
        }
    
    def _extract_company_name(self, main_category: Dict[str, Any]) -> Optional[str]:
        """
        Extract company name from the main category data.
        
        Args:
            main_category: The main category data
            
        Returns:
            Company name or None if not found
        """
        # Try to extract from company_info
        if "company_info" in main_category and main_category["company_info"]:
            company_info = main_category["company_info"]
            if "company_name" in company_info and company_info["company_name"]:
                return company_info["company_name"]
        
        # Try to extract from direct fields in main_category
        if "company_name" in main_category and main_category["company_name"]:
            return main_category["company_name"]
            
        if "business_name" in main_category and main_category["business_name"]:
            return main_category["business_name"]
            
        # Check for common company information fields or extracted_text
        if "extracted_text" in main_category:
            # Try to extract company name from the extracted text using LLM
            return self._extract_company_name_from_text(main_category["extracted_text"])
            
        # Try looking at any other fields that might contain company info
        for key, value in main_category.items():
            if isinstance(value, dict) and "company" in key.lower():
                if "name" in value:
                    return value["name"]
                elif "company_name" in value:
                    return value["company_name"]
                elif "business_name" in value:
                    return value["business_name"]
        
        return None
        
    def _extract_company_name_from_text(self, text: str) -> Optional[str]:
        """
        Use LLM to extract company name from text.
        
        Args:
            text: Text that might contain company name
            
        Returns:
            Company name or None
        """
        try:
            prompt = f"""
            Extract the company name from the following text. Return ONLY the company name, nothing else.
            If you can't find a specific company name, use contextual information to determine the likely company name.
            
            Text:
            {text}
            
            Company name:
            """
            
            chain = LLMChain(llm=self.llm, prompt=ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a business data analyst who specializes in extracting company names from documents."),
                ("user", prompt)
            ]))
            
            result = chain.invoke({})
            if result and "text" in result:
                company_name = result["text"].strip()
                # Check if it's not just an empty string or generic response
                if company_name and not any(x in company_name.lower() for x in ["unknown", "not found", "unable", "cannot", "no company"]):
                    return company_name
            return None
        except Exception as e:
            print(f"Error extracting company name from text: {str(e)}")
            return None
    
    def _enhance_main_category(self, main_category: Dict[str, Any], company_name: str) -> Dict[str, Any]:
        """
        Enhance main category data with web search results.
        
        Args:
            main_category: The main category data from PDF agent
            company_name: The name of the company
            
        Returns:
            Enhanced main category data
        """
        # Create a copy to avoid modifying the original
        enhanced = main_category.copy()
        
        # Check if the data structure follows our expected format
        if "company_info" not in enhanced:
            # Create standard structure if not present
            enhanced["company_info"] = {}
            
            # Try to move relevant fields to company_info
            for field in ["business_name", "year_of_founding", "location_of_headquarters", 
                         "industry", "business_model", "employees", "website_link", 
                         "one_sentence_pitch"]:
                if field in enhanced:
                    enhanced["company_info"][field.replace("business_name", "company_name")] = enhanced.pop(field)
        
        # First, check what data is missing in the company info
        company_info = enhanced.get("company_info", {})
        missing_company_fields = [
            field for field in [
                "year_of_founding", "location_of_headquarters", "industry",
                "business_model", "employees", "website_link", "one_sentence_pitch"
            ] if field not in company_info or not company_info.get(field)
        ]
        
        # Check which financial metrics are missing
        missing_financial_fields = [
            field for field in [
                "annual_recurring_revenue", "monthly_recurring_revenue", 
                "customer_acquisition_cost", "customer_lifetime_value",
                "cltv_cac_ratio", "gross_margin", "revenue_growth_rate_yoy",
                "revenue_growth_rate_mom"
            ] if field not in enhanced or not enhanced.get(field)
        ]
        
        # If we have missing company info, search for it
        if missing_company_fields:
            try:
                company_data = WebSearchUtils.search_company_info(company_name)
                for field in missing_company_fields:
                    if field in company_data and company_data[field]:
                        enhanced["company_info"][field] = company_data[field]
            except Exception as e:
                print(f"Error retrieving company info: {str(e)}")
        
        # If we have missing financial metrics, search for them
        if missing_financial_fields:
            try:
                financial_data = WebSearchUtils.search_financial_data(company_name)
                for field in missing_financial_fields:
                    if field in financial_data and financial_data[field]:
                        enhanced[field] = financial_data[field]
            except Exception as e:
                print(f"Error retrieving financial data: {str(e)}")
        
        # If we have a website but no social profiles, try to get them
        if enhanced["company_info"].get("website_link") and not enhanced["company_info"].get("linkedin_profile_ceo"):
            try:
                social_data = WebSearchUtils.extract_social_profiles(enhanced["company_info"]["website_link"])
                if "ceo_linkedin" in social_data and social_data["ceo_linkedin"]:
                    enhanced["company_info"]["linkedin_profile_ceo"] = social_data["ceo_linkedin"]
            except Exception as e:
                print(f"Error retrieving social profiles: {str(e)}")
        
        # Extract and structure any additional information from extracted_text if present
        if "extracted_text" in enhanced:
            try:
                enhanced = self._extract_additional_info_from_text(enhanced, enhanced["extracted_text"])
            except Exception as e:
                print(f"Error extracting additional info from text: {str(e)}")
        
        # Run a final integration check with LLM
        enhanced = self._integrate_data_with_llm(enhanced, company_name)
        
        return enhanced
        
    def _extract_additional_info_from_text(self, current_data: Dict[str, Any], text: str) -> Dict[str, Any]:
        """
        Extract additional structured information from text content.
        
        Args:
            current_data: Current data structure
            text: The text to extract information from
            
        Returns:
            Enhanced data with additional extracted information
        """
        try:
            # Create a copy of the current data
            enhanced = current_data.copy()
            
            # Try to parse any JSON in the text
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', text)
            extracted_json = {}
            
            if json_match:
                try:
                    json_str = json_match.group(1) or json_match.group(2)
                    extracted_json = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # Use LLM to extract structured data if JSON parsing fails or to enhance it
            if not extracted_json:
                prompt = f"""
                Extract all structured information from this text and return it as JSON:
                
                {text}
                
                Focus on extracting company information, financial metrics, operational data, 
                product details, and any other relevant business information.
                """
                
                chain = LLMChain(llm=self.llm, prompt=ChatPromptTemplate.from_messages([
                    SystemMessage(content="You are a data extraction specialist who can extract structured information from text."),
                    ("user", prompt)
                ]))
                
                result = chain.invoke({})
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', result["text"])
                if json_match:
                    try:
                        json_str = json_match.group(1) or json_match.group(2)
                        extracted_json = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            
            # If we extracted any JSON, try to integrate it with our current data
            if extracted_json:
                # Handle company information
                if "company_information" in extracted_json or "company_info" in extracted_json:
                    company_data = extracted_json.get("company_information", {}) or extracted_json.get("company_info", {})
                    for key, value in company_data.items():
                        if value and key != "address":  # Skip address for now
                            # Map common field names
                            mapped_key = {
                                "website": "website_link",
                                "phone": "phone_number",
                                "email": "contact_email"
                            }.get(key, key)
                            
                            if mapped_key not in enhanced["company_info"] or not enhanced["company_info"][mapped_key]:
                                enhanced["company_info"][mapped_key] = value
                
                # Handle financial metrics
                if "financial_metrics" in extracted_json:
                    financial_data = extracted_json["financial_metrics"]
                    # Try to extract any numeric values
                    for key, value in financial_data.items():
                        if isinstance(value, (int, float)) or (isinstance(value, str) and any(c.isdigit() for c in value)):
                            # Try to clean and map the key
                            clean_key = key.lower().replace(" ", "_")
                            if "revenue" in clean_key and clean_key not in enhanced:
                                enhanced["annual_recurring_revenue"] = self._extract_numeric_value(value)
                
                # Handle operational data
                if "operational_data" in extracted_json:
                    op_data = extracted_json["operational_data"]
                    
                    # Extract market size if available
                    if "market_size" in op_data and op_data["market_size"]:
                        if "market_size" not in enhanced:
                            enhanced["market_size"] = op_data["market_size"]
                    
                    # Extract user counts if available
                    if "market_validation" in op_data and isinstance(op_data["market_validation"], dict):
                        validation = op_data["market_validation"]
                        if "total_users" in validation and "monthly_active_users" not in enhanced:
                            enhanced["monthly_active_users"] = self._extract_numeric_value(validation.get("total_users", "0"))
                        if "active_users" in validation and "monthly_active_users" not in enhanced:
                            enhanced["monthly_active_users"] = self._extract_numeric_value(validation.get("active_users", "0"))
                    
                    # Extract business model info
                    if "business_model" in op_data and isinstance(op_data["business_model"], dict):
                        model = op_data["business_model"]
                        if "commission_rate" in model and "commission_rate" not in enhanced:
                            enhanced["commission_rate"] = model["commission_rate"]
            
            # Remove the extracted_text field to clean up the final output
            if "extracted_text" in enhanced:
                del enhanced["extracted_text"]
                
            return enhanced
            
        except Exception as e:
            print(f"Error in _extract_additional_info_from_text: {str(e)}")
            return current_data
            
    def _extract_numeric_value(self, value_str: Any) -> Optional[int]:
        """
        Extract a numeric value from a string or other value.
        
        Args:
            value_str: The value to extract a number from
            
        Returns:
            Extracted integer or None
        """
        if isinstance(value_str, (int, float)):
            return int(value_str)
            
        if not isinstance(value_str, str):
            return None
            
        # Try to extract numbers
        import re
        # Remove commas and find numbers
        clean_str = value_str.replace(',', '')
        match = re.search(r'(\d+)', clean_str)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None
    
    def _enhance_search_category(self, search_category: Dict[str, Any], company_name: str) -> Dict[str, Any]:
        """
        Enhance or create search category data using web search.
        
        Args:
            search_category: Existing search category data or empty dict
            company_name: The name of the company
            
        Returns:
            Enhanced search category data
        """
        # If search category already has data, return it
        if search_category and any(search_category.values()):
            return search_category
        
        # Otherwise, search for new data
        try:
            search_data = WebSearchUtils.search_category_to_search_data(company_name)
            return search_data
        except Exception as e:
            print(f"Error enhancing search category: {str(e)}")
            return search_category
    
    def _integrate_data_with_llm(self, data: Dict[str, Any], company_name: str) -> Dict[str, Any]:
        """
        Use an LLM to integrate and validate the enhanced data.
        This helps ensure the data is consistent and makes sense.
        
        Args:
            data: The enhanced data
            company_name: The name of the company
            
        Returns:
            Integrated and validated data
        """
        try:
            # Create a prompt for the LLM
            prompt = f"""
            I have collected the following information about {company_name}:
            
            {json.dumps(data, indent=2)}
            
            Please review this information and fix any inconsistencies or errors.
            If any values seem unrealistic or don't make sense, correct them or remove them.
            
            Return the corrected data as a valid JSON object that matches the original structure.
            """
            
            # Create a structured prompt for the LLM
            chain = LLMChain(llm=self.llm, prompt=ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a financial data analyst who specializes in startup metrics. You can identify inconsistencies in financial data and fix them."),
                ("user", prompt)
            ]))
            
            # Get the response
            result = chain.invoke({})
            content = result["text"]
            
            # Try to extract the JSON from the response
            import re
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', content)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                return json.loads(json_str)
            
            # If no JSON found, return the original data
            return data
        except Exception as e:
            print(f"Error integrating data with LLM: {str(e)}")
            return data