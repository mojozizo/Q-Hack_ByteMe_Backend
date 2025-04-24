import json
import os
import time
import requests
import re
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
from etl.agent.financial_agent import FinancialAgent
from etl.extract.abstract_extracter import AbstractExtracter
from etl.util.file_util import create_or_get_upload_folder
from models.model import StartupMetrics
from models.news_model import NewsModel


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
        self.financial_agent = FinancialAgent()
        
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
        financial_data = {}
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
                
            # Step 5: Get financial data using the financial agent
            try:
                print(f"Getting financial data for company: {company_name}")
                start_time = time.time()
                financial_data_json = self.financial_agent._run(company_name)
                if financial_data_json:
                    try:
                        financial_data = json.loads(financial_data_json)
                        print(f"Financial data extraction completed in {time.time() - start_time:.2f} seconds")
                        
                        # Add the financial data to the web_enhanced_results
                        if "main_category" in web_enhanced_results:
                            # Fill missing financial metrics from financial data
                            self._fill_missing_financial_metrics(web_enhanced_results["main_category"], financial_data)
                            
                        # Save financial data separately too
                        web_enhanced_results["financial_data"] = financial_data
                    except json.JSONDecodeError:
                        error_details["financial_parse_error"] = "Failed to parse financial data JSON"
                        print("Error parsing financial data JSON")
                else:
                    error_details["financial_error"] = "No financial data received"
                    print("No financial data received from financial agent")
            except Exception as e:
                error_details["financial_error"] = str(e)
                print(f"Error retrieving financial data: {str(e)}")
                financial_data = {"error": f"Financial data error: {str(e)}"}
            
            # Step 6: Get LinkedIn data if CEO LinkedIn profile is available
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
            
            # Step 7: Get news data about the company
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
            
            # Step 8: Integrate all data sources
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
                    "financial_data": financial_data,
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
                "financial_data": financial_data,
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
        
        # Process LinkedIn data to extract founder metrics
        founder_metrics = self._extract_founder_linkedin_data(linkedin_data)
        
        # Process news data to extract risk assessments
        risk_assessment = self._process_news_for_risks(news_data)
        
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
        
        # Update the main_category with founder metrics and risk assessment data
        if "main_category" in consolidated:
            # Add founder metrics from LinkedIn
            for key, value in founder_metrics.items():
                if value is not None:
                    consolidated["main_category"][key] = value
            
            # Add risk assessment from news
            for key, value in risk_assessment.items():
                consolidated["main_category"][key] = value
        
        # Do the same for the metrics field if it exists (new format)
        if "metrics" in consolidated:
            # Add founder metrics from LinkedIn
            for key, value in founder_metrics.items():
                if value is not None:
                    consolidated["metrics"][key] = value
            
            # Add risk assessment from news
            for key, value in risk_assessment.items():
                consolidated["metrics"][key] = value
        
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

    def _fill_missing_financial_metrics(self, main_category: Dict[str, Any], financial_data: Dict[str, Any]) -> None:
        """
        Fill missing financial metrics in the main category data with data from the financial agent.
        
        Args:
            main_category: The main category data to update
            financial_data: Financial data from the financial agent
        """
        if not financial_data or not main_category:
            return
            
        # Map financial agent data fields to main_category fields
        financial_mapping = {
            "revenue": "annual_recurring_revenue",
            "net_income": "net_income",
            "total_assets": "total_assets",
            "total_liabilities": "total_liabilities",
            "market_cap": "market_cap"
        }
        
        # Fill in missing financial metrics
        for fin_field, main_field in financial_mapping.items():
            if fin_field in financial_data and financial_data[fin_field]:
                # Only fill if the field doesn't exist or is empty in main_category
                if main_field not in main_category or not main_category[main_field]:
                    # Try to convert to integer if the value is numeric
                    try:
                        # Remove non-numeric characters (like $ and ,)
                        value_str = str(financial_data[fin_field])
                        clean_value = ''.join(c for c in value_str if c.isdigit() or c == '.')
                        if clean_value:
                            if '.' in clean_value:
                                main_category[main_field] = float(clean_value)
                            else:
                                main_category[main_field] = int(clean_value)
                        else:
                            main_category[main_field] = financial_data[fin_field]
                    except (ValueError, TypeError):
                        # If conversion fails, use the original value
                        main_category[main_field] = financial_data[fin_field]
        
        # Add company information if available
        if "company_info" not in main_category:
            main_category["company_info"] = {}
            
        company_info = main_category["company_info"]
        
        # Fill in company info fields from financial data
        if "company_name" in financial_data and financial_data["company_name"] and "company_name" not in company_info:
            company_info["company_name"] = financial_data["company_name"]
            
        if "sic_description" in financial_data and financial_data["sic_description"] and "industry" not in company_info:
            company_info["industry"] = financial_data["sic_description"]
            
        if "fiscal_year" in financial_data and financial_data["fiscal_year"] and "year_of_founding" not in company_info:
            try:
                fiscal_year = financial_data["fiscal_year"]
                # Only use this as founding year if it's a reasonable guess (not too recent)
                year_value = int(fiscal_year)
                current_year = 2025  # As of the context date
                if year_value < current_year - 2:  # More than 2 years ago could be founding year
                    company_info["year_of_founding"] = year_value
            except (ValueError, TypeError):
                pass
                
        # Add financial summary if available
        if "financial_summary" in financial_data and financial_data["financial_summary"] and "pitch_deck_summary" not in company_info:
            company_info["financial_summary"] = financial_data["financial_summary"]

    def _process_news_for_risks(self, news_data: Union[Dict[str, Any], str, NewsModel]) -> Dict[str, Any]:
        """
        Process news data to extract regulatory and trend risks.
        
        Args:
            news_data: News data from the news agent, could be a dictionary, JSON string,
                      or a NewsModel instance
                
        Returns:
            Dictionary containing risk assessment fields
        """
        risk_assessment = {
            "regulatory_risks": False,
            "trend_risks": False,
            "news_sentiment": "neutral",
            "recent_news_summary": ""
        }
        
        try:
            # Handle different input types
            if isinstance(news_data, str):
                try:
                    news_data = json.loads(news_data)
                except json.JSONDecodeError:
                    print("Could not parse news_data as JSON")
                    return risk_assessment
            
            if isinstance(news_data, NewsModel):
                # Access the model attributes directly
                news_content = {
                    "summary": news_data.summary,
                    "title": news_data.title,
                    "description": news_data.description,
                    "tone": news_data.tone,
                    "keywords": news_data.keywords
                }
            elif isinstance(news_data, dict):
                news_content = news_data
            else:
                print(f"Unexpected news_data type: {type(news_data)}")
                return risk_assessment
            
            # Extract the news summary for later use
            summary = news_content.get("summary", "")
            if summary:
                risk_assessment["recent_news_summary"] = summary
            
            # Look for regulatory keywords in the news content
            regulatory_keywords = [
                "regulation", "compliance", "law", "legal", "legislation", 
                "regulatory", "regulator", "fine", "penalty", "sanction",
                "investigation", "lawsuit", "litigation", "court", "antitrust"
            ]
            
            # Look for trend risk keywords
            trend_risk_keywords = [
                "disrupt", "disruption", "obsolete", "obsolescence", "decline",
                "decline in demand", "market shift", "changing market", "trend change",
                "technological shift", "innovation challenge", "market shrink"
            ]
            
            # Extract content to analyze
            keywords = news_content.get("keywords", [])
            title = news_content.get("title", "")
            description = news_content.get("description", "")
            full_text = f"{title} {description} {summary}"
            
            # Check for regulatory risks
            if any(keyword.lower() in full_text.lower() for keyword in regulatory_keywords):
                risk_assessment["regulatory_risks"] = True
            
            if any(keyword.lower() in keywords for keyword in regulatory_keywords):
                risk_assessment["regulatory_risks"] = True
            
            # Check for trend risks
            if any(keyword.lower() in full_text.lower() for keyword in trend_risk_keywords):
                risk_assessment["trend_risks"] = True
                
            if any(keyword.lower() in keywords for keyword in trend_risk_keywords):
                risk_assessment["trend_risks"] = True
            
            # Determine sentiment from tone
            tone = news_content.get("tone", "").lower()
            if "positive" in tone:
                risk_assessment["news_sentiment"] = "positive"
            elif "negative" in tone:
                risk_assessment["news_sentiment"] = "negative"
            
            # If we need more sophisticated analysis, we could use the LLM here
            if not risk_assessment["regulatory_risks"] and not risk_assessment["trend_risks"]:
                # Use LLM to analyze the text for less obvious risk indicators
                prompt = f"""
                Analyze this news content about a company and determine if there are any regulatory or trend risks.
                Regulatory risks involve legal, compliance, or regulatory challenges the company might face.
                Trend risks involve market shifts, changing demands, or technological disruptions that could affect the company.
                
                News content: {full_text}
                
                Return ONLY a valid JSON object with these fields:
                - regulatory_risks: true/false
                - trend_risks: true/false
                - reasoning: brief explanation
                """
                
                response = self.llm.invoke(prompt)
                content = response.content
                
                try:
                    result = json.loads(content)
                    risk_assessment["regulatory_risks"] = result.get("regulatory_risks", False)
                    risk_assessment["trend_risks"] = result.get("trend_risks", False)
                except json.JSONDecodeError:
                    # Try to extract JSON with regex
                    json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', content)
                    if json_match:
                        json_str = json_match.group(1) or json_match.group(2)
                        try:
                            result = json.loads(json_str)
                            risk_assessment["regulatory_risks"] = result.get("regulatory_risks", False)
                            risk_assessment["trend_risks"] = result.get("trend_risks", False)
                        except json.JSONDecodeError:
                            pass
            
            return risk_assessment
            
        except Exception as e:
            print(f"Error processing news for risks: {str(e)}")
            return risk_assessment

    def _extract_founder_linkedin_data(self, linkedin_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Extract founder metrics from LinkedIn data.
        
        Args:
            linkedin_data: LinkedIn data from the LinkedIn agent, could be a dictionary or JSON string
            
        Returns:
            Dictionary containing founder metrics fields
        """
        founder_metrics = {
            "founder_industry_experience": None,
            "founder_past_exits": None,
            "founder_background": None,
            "founder_linkedin_url": None,
            "founder_linkedin_summary": None,
            "founder_skills": None
        }
        
        try:
            # Handle different input types
            if isinstance(linkedin_data, str):
                try:
                    linkedin_data = json.loads(linkedin_data)
                except json.JSONDecodeError:
                    print("Could not parse linkedin_data as JSON")
                    return founder_metrics
            
            if not isinstance(linkedin_data, dict):
                print(f"Unexpected linkedin_data type: {type(linkedin_data)}")
                return founder_metrics
            
            # Extract summary if available
            if "summary" in linkedin_data and linkedin_data["summary"]:
                founder_metrics["founder_linkedin_summary"] = linkedin_data["summary"]
            
            # Extract skills if available
            if "skills" in linkedin_data and linkedin_data["skills"]:
                founder_metrics["founder_skills"] = linkedin_data["skills"]
            
            # Extract URL from source data if available
            if "source_url" in linkedin_data and linkedin_data["source_url"]:
                founder_metrics["founder_linkedin_url"] = linkedin_data["source_url"]
            
            # For more advanced metrics, we need to analyze the data with the LLM
            summary = linkedin_data.get("summary", "")
            skills = linkedin_data.get("skills", [])
            title = linkedin_data.get("title", "")
            name = linkedin_data.get("name", "")
            current_company = linkedin_data.get("current_company", "")
            experiences = linkedin_data.get("experiences", [])
            education = linkedin_data.get("education", [])
            
            # Convert experiences and education to strings if they're lists or dicts
            if isinstance(experiences, (list, dict)):
                experiences = json.dumps(experiences)
            
            if isinstance(education, (list, dict)):
                education = json.dumps(education)
            
            # Use LLM to analyze LinkedIn data for founder metrics
            prompt = f"""
            Analyze this LinkedIn profile data and extract the following founder metrics:
            
            1. Founder industry experience (as an integer 1-5 or in years)
            2. Number of past exits (count of successful exits as an integer)
            3. Founder background quality score (scale 1-5, based on education and experience at top companies/universities)
            
            Profile data:
            - Name: {name}
            - Title: {title}
            - Current company: {current_company}
            - Skills: {', '.join(skills) if isinstance(skills, list) else skills}
            - Summary: {summary}
            - Experiences: {experiences}
            - Education: {education}
            
            Return ONLY a valid JSON object with these fields:
            - founder_industry_experience: integer (years of experience or score 1-5)
            - founder_past_exits: integer (count of exits)
            - founder_background: integer (score 1-5)
            - reasoning: brief explanation for each score
            """
            
            response = self.llm.invoke(prompt)
            content = response.content
            
            try:
                result = json.loads(content)
                # Update the metrics with the LLM analysis
                founder_metrics["founder_industry_experience"] = result.get("founder_industry_experience")
                founder_metrics["founder_past_exits"] = result.get("founder_past_exits")
                founder_metrics["founder_background"] = result.get("founder_background")
            except json.JSONDecodeError:
                # Try to extract JSON with regex
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', content)
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                    try:
                        result = json.loads(json_str)
                        founder_metrics["founder_industry_experience"] = result.get("founder_industry_experience")
                        founder_metrics["founder_past_exits"] = result.get("founder_past_exits")
                        founder_metrics["founder_background"] = result.get("founder_background")
                    except json.JSONDecodeError:
                        pass
            
            return founder_metrics
            
        except Exception as e:
            print(f"Error extracting founder LinkedIn data: {str(e)}")
            return founder_metrics