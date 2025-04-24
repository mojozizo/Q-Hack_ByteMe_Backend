import json
import os
import time
import requests
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from fastapi import UploadFile

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from etl.agent.pdf_agent import PDFAgentExecutor
from etl.agent.web_search_agent import WebSearchAgent
from etl.agent.linkedin_agent import LinkedInAgent
from etl.agent.news_agent import NewsAgent
from etl.extract.abstract_extracter import AbstractExtracter
from etl.util.file_util import create_or_get_upload_folder
from models.model import StartupMetrics


class OrchestratorAgent(AbstractExtracter):
    """
    Main orchestrator agent that coordinates between specialized agents.
    This agent manages the workflow between PDF, LinkedIn, Web Search, and News agents.
    """

    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the orchestrator agent.
        
        Args:
            model_name: The OpenAI model to use for integration and analysis
        """
        super().__init__()
        self.model_name = model_name
        
        # Initialize specialized agents
        self.pdf_agent = PDFAgentExecutor(model_name=model_name)
        self.web_search_agent = WebSearchAgent(model_name=model_name)
        self.linkedin_agent = LinkedInAgent()
        self.news_agent = NewsAgent()
        
        # LLM for integration tasks
        self.llm = ChatOpenAI(temperature=0.3, model=model_name)
        
    def extract(self, file: UploadFile, query: str = None) -> str:
        """
        Orchestrate the extraction process using multiple specialized agents.
        
        Args:
            file: The uploaded file to process (typically a PDF)
            query: Optional query to guide the extraction
            
        Returns:
            JSON string containing the consolidated results
        """
        pdf_results = {}
        web_enhanced_results = {}
        linkedin_data = {}
        news_data = {}
        error_details = {}
        
        try:
            # Step 1: Save the uploaded file
            try:
                upload_dir = create_or_get_upload_folder()
                file_path = Path(upload_dir) / file.filename
                
                with open(file_path, "wb") as f:
                    contents = file.file.read()
                    f.write(contents)
                    
                print(f"File saved successfully at: {file_path}")
            except Exception as e:
                error_details["file_save_error"] = str(e)
                raise Exception(f"Error saving file: {str(e)}")
                
            # Step 2: Extract text content from PDF
            try:
                pdf_text = self._extract_text_from_pdf(file_path)
                print(f"Successfully extracted {len(pdf_text)} characters from PDF")
            except Exception as e:
                error_details["pdf_extraction_error"] = str(e)
                raise Exception(f"Error extracting text from PDF: {str(e)}")
            
            # Step 3: Use PDF agent to extract structured data
            try:
                print("Extracting data using PDF agent...")
                start_time = time.time()
                pdf_results = self.pdf_agent.extract_from_pdf_text(pdf_text, enable_web_enrichment=False)
                print(f"PDF agent processing completed in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                error_details["pdf_agent_error"] = str(e)
                print(f"Error in PDF agent: {str(e)}")
                # Continue with partial results instead of failing completely
                pdf_results = {"error": f"PDF agent error: {str(e)}"}
            
            # Step 4: Enhance with web search
            try:
                print("Enhancing data with web search...")
                start_time = time.time()
                web_enhanced_results = self.web_search_agent.enhance_results(pdf_results)
                print(f"Web search enhancement completed in {time.time() - start_time:.2f} seconds")
            except Exception as e:
                error_details["web_search_error"] = str(e)
                print(f"Error in web search: {str(e)}")
                # Use PDF results as fallback
                web_enhanced_results = pdf_results
                web_enhanced_results["web_search_error"] = str(e)
            
            # Extract company name for further processing
            company_name = self._extract_company_name(web_enhanced_results)
            
            if not company_name:
                print("No company name found, cannot proceed with LinkedIn and News extraction")
                web_enhanced_results["warning"] = "No company name found, LinkedIn and News data could not be retrieved"
                return json.dumps(web_enhanced_results, indent=2)
            
            # Step 5: Get LinkedIn data if CEO LinkedIn profile is available
            linkedin_profile = self._extract_linkedin_profile(web_enhanced_results)
            
            if linkedin_profile:
                try:
                    print(f"Extracting LinkedIn data from profile: {linkedin_profile}")
                    start_time = time.time()
                    linkedin_data = json.loads(self.linkedin_agent._run(linkedin_profile))
                    print(f"LinkedIn data extraction completed in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    error_details["linkedin_error"] = str(e)
                    print(f"Error processing LinkedIn data: {str(e)}")
                    linkedin_data = {"error": f"LinkedIn data error: {str(e)}"}
            
            # Step 6: Get news data about the company
            try:
                print(f"Searching for news about: {company_name}")
                start_time = time.time()
                news_data = self.news_agent._run(company_name)
                print(f"News data retrieval completed in {time.time() - start_time:.2f} seconds")
            except requests.exceptions.ConnectionError as e:
                error_details["news_connection_error"] = str(e)
                print(f"Connection error retrieving news data: {str(e)}")
                news_data = {"error": f"News API connection error: {str(e)}"}
            except Exception as e:
                error_details["news_error"] = str(e)
                print(f"Error retrieving news data: {str(e)}")
                news_data = {"error": f"News data error: {str(e)}"}
            
            # Step 7: Integrate all data sources
            try:
                print("Integrating all data sources...")
                consolidated_results = self._integrate_results(
                    web_enhanced_results, 
                    linkedin_data, 
                    news_data
                )
                
                # Add any errors that occurred along the way
                if error_details:
                    consolidated_results["error_details"] = error_details
                
                # Return the final consolidated results
                return json.dumps(consolidated_results, indent=2)
            except Exception as e:
                error_details["integration_error"] = str(e)
                print(f"Error integrating results: {str(e)}")
                
                # Return a fallback response with all the data we have so far
                fallback_results = {
                    "pdf_data": pdf_results,
                    "web_data": web_enhanced_results,
                    "linkedin_data": linkedin_data,
                    "news_data": news_data,
                    "error_details": error_details,
                    "error": "Error during results integration"
                }
                return json.dumps(fallback_results, indent=2)
                
        except Exception as e:
            import traceback
            error_msg = f"Error in orchestrator extraction: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            
            # Return a detailed error response
            error_response = {
                "error": error_msg,
                "error_details": error_details,
                "pdf_results": pdf_results,
                "web_results": web_enhanced_results,
                "linkedin_data": linkedin_data,
                "news_data": news_data
            }
            return json.dumps(error_response, indent=2)
        finally:
            file.file.close()
            
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        import PyPDF2
        
        with open(file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
                
        return text
    
    def _extract_company_name(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Extract company name from the results.
        
        Args:
            results: Results from previous extraction steps
            
        Returns:
            Company name or None if not found
        """
        # Try to extract from main_category.company_info
        if "main_category" in results and results["main_category"]:
            if "company_info" in results["main_category"] and results["main_category"]["company_info"]:
                company_info = results["main_category"]["company_info"]
                if "company_name" in company_info and company_info["company_name"]:
                    return company_info["company_name"]
        
        # Try to extract from metrics if present
        if "metrics" in results and results["metrics"]:
            if "company_name" in results["metrics"]:
                return results["metrics"]["company_name"]
                
        return None
    
    def _extract_linkedin_profile(self, results: Dict[str, Any]) -> Optional[str]:
        """
        Extract LinkedIn profile URL from results.
        
        Args:
            results: Results from previous extraction steps
            
        Returns:
            LinkedIn profile URL or None if not found
        """
        # Try to extract from main_category.company_info
        if "main_category" in results and results["main_category"]:
            if "company_info" in results["main_category"] and results["main_category"]["company_info"]:
                company_info = results["main_category"]["company_info"]
                if "linkedin_profile_ceo" in company_info and company_info["linkedin_profile_ceo"]:
                    return company_info["linkedin_profile_ceo"]
        
        return None
    
    def _integrate_results(
        self, 
        web_results: Dict[str, Any], 
        linkedin_data: Dict[str, Any], 
        news_data: Any
    ) -> Dict[str, Any]:
        """
        Integrate results from all agents into a consolidated structure.
        
        Args:
            web_results: Results from web search agent
            linkedin_data: Results from LinkedIn agent
            news_data: Results from news agent
            
        Returns:
            Consolidated results
        """
        # Create a copy of web results as our base
        consolidated = web_results.copy()
        
        # Add LinkedIn data if available
        if linkedin_data:
            try:
                if isinstance(linkedin_data, str):
                    # Try to parse as JSON if it's a string
                    consolidated["linkedin_data"] = json.loads(linkedin_data)
                elif isinstance(linkedin_data, dict):
                    # Use dictionary directly
                    consolidated["linkedin_data"] = linkedin_data
                else:
                    # For other object types, attempt to convert to dict if possible
                    consolidated["linkedin_data"] = {"raw_data": str(linkedin_data)}
            except json.JSONDecodeError:
                # If it can't be parsed as JSON, store as is
                consolidated["linkedin_data"] = {"raw_data": linkedin_data}
        
        # Add news data if available
        if news_data:
            try:
                if isinstance(news_data, str):
                    # Try to parse as JSON if it's a string
                    consolidated["news_data"] = json.loads(news_data)
                elif hasattr(news_data, "model_dump"):
                    # Handle Pydantic model
                    consolidated["news_data"] = news_data.model_dump()
                elif hasattr(news_data, "dict"):
                    # For older Pydantic versions
                    consolidated["news_data"] = news_data.dict()
                else:
                    # Otherwise use the object directly
                    consolidated["news_data"] = news_data
            except json.JSONDecodeError:
                # If it can't be parsed as JSON, store as is
                consolidated["news_data"] = {"raw_news": news_data}
        
        # Use LLM to provide a final analysis and summary
        consolidated["analysis"] = self._generate_analysis(consolidated)
        
        # Mark as processed by orchestrator
        consolidated["source"] = "orchestrator_agent"
        
        return consolidated
    
    def _generate_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis of the collected data.
        
        Args:
            data: Consolidated data from all agents
            
        Returns:
            Analysis and summary of the data
        """
        try:
            # Create a prompt for the LLM
            prompt = f"""
            Analyze the following consolidated data about a company:
            
            {json.dumps(data, indent=2)}
            
            Please provide:
            1. A concise executive summary (2-3 sentences)
            2. Key strengths of the company
            3. Potential risks or weaknesses
            4. Investment recommendation (on a scale of 1-5, with 5 being highest)
            5. Justification for the recommendation
            
            Return ONLY a valid JSON object with these fields: executive_summary, strengths (array), 
            weaknesses (array), investment_score (number 1-5), justification
            """
            
            # Create a structured prompt for the LLM
            chain = LLMChain(llm=self.llm, prompt=ChatPromptTemplate.from_messages([
                SystemMessage(content="You are an expert venture capital analyst who specializes in startup evaluation."),
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
            
            # If no JSON found, create a simple structure
            return {
                "executive_summary": "Analysis could not be generated in the expected format.",
                "raw_analysis": content
            }
        except Exception as e:
            print(f"Error generating analysis: {str(e)}")
            return {
                "error": f"Error generating analysis: {str(e)}"
            }