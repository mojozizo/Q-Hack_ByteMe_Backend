import os
import time
import requests
from typing import Dict, Any, Optional, Type, List
from pathlib import Path
import json

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from models.model import Category, CompanyInfo, CategoryToSearch, StartupMetrics
from etl.util.web_search_util import WebSearchUtils
from etl.util.model_util import enrich_category_to_search, enrich_model_from_web


class PDFAgentTools:
    """Collection of tools for the PDF extraction agent workflow."""
    
    @tool
    def extract_company_info(pdf_text: str) -> Dict[str, Any]:
        """
        Extract company information from PDF text.
        
        Args:
            pdf_text: The text content of the PDF
            
        Returns:
            Dictionary with company information fields
        """
        try:
            llm = ChatOpenAI(temperature=0, model="gpt-4o")
            
            prompt = """
            Extract the following company information from the provided text:
            - Company Name
            - Official Company Name (if different)
            - Year of Founding (as integer)
            - Location of Headquarters
            - Business Model
            - Industry
            - Required Funding Amount (as integer)
            - Number of Employees (e.g., "10-50")
            - Website Link
            - One Sentence Pitch
            - LinkedIn Profile of CEO
            
            Provide a detailed summary of the document highlighting all important aspects of the company.
            
            Return a valid JSON object with these keys:
            - company_name
            - official_company_name
            - year_of_founding
            - location_of_headquarters
            - business_model
            - industry
            - required_funding_amount
            - employees
            - website_link
            - one_sentence_pitch
            - linkedin_profile_ceo
            - pitch_deck_summary
            
            Only include fields where you can find information in the text.
            """
            
            chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([
                SystemMessage(content="You are a specialized financial document analyzer focused on extracting company information from pitch decks."),
                ("user", prompt + "\n\nDocument text:\n{text}")
            ]))
            
            result = chain.invoke({"text": pdf_text})
            try:
                # Try to parse the result as JSON
                start_idx = result["text"].find("{")
                end_idx = result["text"].rfind("}") + 1
                json_str = result["text"][start_idx:end_idx]
                return json.loads(json_str)
            except:
                # Return an empty dict if parsing fails
                return {}
        except requests.exceptions.RequestException as e:
            # Handle connection errors
            print(f"Connection error in extract_company_info: {str(e)}")
            # Provide basic fallback data
            return {
                "error": f"Connection error: {str(e)}",
                "company_name": "Unknown",
                "pitch_deck_summary": "Could not extract summary due to connection error"
            }
        except Exception as e:
            print(f"Error in extract_company_info: {str(e)}")
            return {
                "error": f"Error: {str(e)}"
            }
    
    @tool
    def extract_financial_metrics(pdf_text: str) -> Dict[str, Any]:
        """
        Extract financial metrics from PDF text.
        
        Args:
            pdf_text: The text content of the PDF
            
        Returns:
            Dictionary with financial metrics fields
        """
        llm = ChatOpenAI(temperature=0, model="gpt-4o")
        
        prompt = """
        Extract the following financial metrics from the provided text:
        - Annual Recurring Revenue (ARR) in USD as an integer
        - Monthly Recurring Revenue (MRR) in USD as an integer
        - Customer Acquisition Cost (CAC) in USD as an integer
        - Customer Lifetime Value (CLTV) in USD as an integer
        - CLTV/CAC Ratio as an integer
        - Gross Margin percentage as an integer (e.g., 75% = 75)
        - Revenue Growth Rate year-over-year as an integer percentage
        - Revenue Growth Rate month-over-month as an integer percentage
        
        Return a valid JSON object with these keys:
        - annual_recurring_revenue
        - monthly_recurring_revenue
        - customer_acquisition_cost
        - customer_lifetime_value
        - cltv_cac_ratio
        - gross_margin
        - revenue_growth_rate_yoy
        - revenue_growth_rate_mom
        
        Only include fields where you can find information in the text. Convert all values to integers.
        """
        
        chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a specialized financial analyst focused on extracting financial metrics from pitch decks."),
            ("user", prompt + "\n\nDocument text:\n{text}")
        ]))
        
        result = chain.invoke({"text": pdf_text})
        try:
            # Try to parse the result as JSON
            start_idx = result["text"].find("{")
            end_idx = result["text"].rfind("}") + 1
            json_str = result["text"][start_idx:end_idx]
            return json.loads(json_str)
        except:
            # Return an empty dict if parsing fails
            return {}
    
    @tool
    def extract_operational_metrics(pdf_text: str) -> Dict[str, Any]:
        """
        Extract operational metrics from PDF text.
        
        Args:
            pdf_text: The text content of the PDF
            
        Returns:
            Dictionary with operational metrics fields
        """
        llm = ChatOpenAI(temperature=0, model="gpt-4o")
        
        prompt = """
        Extract the following operational metrics from the provided text:
        - Sales Cycle Length in days as an integer
        - Monthly Active Users (MAU) as an integer
        - User Growth Rate year-over-year as an integer percentage
        - User Growth Rate month-over-month as an integer percentage
        - Conversion Rate from free to paid as an integer percentage
        
        Return a valid JSON object with these keys:
        - sales_cycle_length
        - monthly_active_users
        - user_growth_rate_yoy
        - user_growth_rate_mom
        - conversion_rate
        
        Only include fields where you can find information in the text. Convert all values to integers.
        """
        
        chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a specialized operations analyst focused on extracting operational metrics from pitch decks."),
            ("user", prompt + "\n\nDocument text:\n{text}")
        ]))
        
        result = chain.invoke({"text": pdf_text})
        try:
            # Try to parse the result as JSON
            start_idx = result["text"].find("{")
            end_idx = result["text"].rfind("}") + 1
            json_str = result["text"][start_idx:end_idx]
            return json.loads(json_str)
        except:
            # Return an empty dict if parsing fails
            return {}
    
    @tool
    def extract_strategic_and_market_metrics(pdf_text: str) -> Dict[str, Any]:
        """
        Extract strategic and market metrics from PDF text.
        
        Args:
            pdf_text: The text content of the PDF
            
        Returns:
            Dictionary with strategic and market metrics fields
        """
        llm = ChatOpenAI(temperature=0, model="gpt-4o")
        
        prompt = """
        Extract the following strategic and market metrics from the provided text:
        - Pricing Strategy Maturity as an integer between 1-5
        - Burn Rate (monthly) in USD as an integer
        - Runway in months as an integer
        - IP Protection (1 for yes, 0 for no)
        - Market Competitiveness as an integer between 1-5
        - Market Timing advantage as an integer between 1-5
        - Cap Table Cleanliness as an integer between 1-5
        
        Return a valid JSON object with these keys:
        - pricing_strategy_maturity
        - burn_rate
        - runway
        - ip_protection
        - market_competitiveness
        - market_timing
        - cap_table_cleanliness
        
        Only include fields where you can find information in the text. Convert all values to integers.
        """
        
        chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a specialized market analyst focused on extracting strategic and market metrics from pitch decks."),
            ("user", prompt + "\n\nDocument text:\n{text}")
        ]))
        
        result = chain.invoke({"text": pdf_text})
        try:
            # Try to parse the result as JSON
            start_idx = result["text"].find("{")
            end_idx = result["text"].rfind("}") + 1
            json_str = result["text"][start_idx:end_idx]
            return json.loads(json_str)
        except:
            # Return an empty dict if parsing fails
            return {}
    
    @tool
    def extract_founder_metrics(pdf_text: str) -> Dict[str, Any]:
        """
        Extract founder and team metrics from PDF text.
        
        Args:
            pdf_text: The text content of the PDF
            
        Returns:
            Dictionary with founder and team metrics fields
        """
        llm = ChatOpenAI(temperature=0, model="gpt-4o")
        
        prompt = """
        Extract the following founder and team metrics from the provided text:
        - Founder Industry Experience as an integer (years or scale 1-5)
        - Founder Past Exits as an integer
        - Founder Background/pedigree as an integer between 1-5
        - Country of Headquarters as a string
        
        Return a valid JSON object with these keys:
        - founder_industry_experience
        - founder_past_exits
        - founder_background
        - country_of_headquarters
        
        Only include fields where you can find information in the text. Convert all values to integers, except for country_of_headquarters which should be a string.
        """
        
        chain = LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a specialized HR analyst focused on extracting founder and team metrics from pitch decks."),
            ("user", prompt + "\n\nDocument text:\n{text}")
        ]))
        
        result = chain.invoke({"text": pdf_text})
        try:
            # Try to parse the result as JSON
            start_idx = result["text"].find("{")
            end_idx = result["text"].rfind("}") + 1
            json_str = result["text"][start_idx:end_idx]
            return json.loads(json_str)
        except:
            # Return an empty dict if parsing fails
            return {}
    
    @tool
    def enrich_with_web_data(company_name: str) -> Dict[str, Any]:
        """
        Enrich extracted data with information from web sources.
        
        Args:
            company_name: The name of the company to search for
            
        Returns:
            Dictionary with company information from web sources
        """
        try:
            # First, get basic company info
            company_data = WebSearchUtils.search_company_info(company_name)
            
            # Then get financial data
            financial_data = WebSearchUtils.search_financial_data(company_name)
            
            # Combine the data
            enriched_data = {**company_data, **financial_data}
            return enriched_data
        except Exception as e:
            print(f"Error enriching with web data: {str(e)}")
            return {}
    
    @tool
    def extract_social_profiles(website_url: str) -> Dict[str, str]:
        """
        Extract social media profiles from a company website.
        
        Args:
            website_url: The URL of the company website
            
        Returns:
            Dictionary mapping social platform names to profile URLs
        """
        try:
            return WebSearchUtils.extract_social_profiles(website_url)
        except Exception as e:
            print(f"Error extracting social profiles: {str(e)}")
            return {}
    
    @tool
    def extract_linkedin_data(profile_url: str) -> Dict[str, Any]:
        """
        Extract data from a LinkedIn profile.
        
        Args:
            profile_url: The URL of the LinkedIn profile
            
        Returns:
            Dictionary with data from the LinkedIn profile
        """
        try:
            return WebSearchUtils.search_linkedin(profile_url=profile_url)
        except Exception as e:
            print(f"Error extracting LinkedIn data: {str(e)}")
            return {}
    
    @tool
    def search_news(company_name: str) -> Dict[str, Any]:
        """
        Search for news articles about a company.
        
        Args:
            company_name: The name of the company to search for
            
        Returns:
            Dictionary with news article data
        """
        try:
            return WebSearchUtils.search_news(company_name)
        except Exception as e:
            print(f"Error searching news: {str(e)}")
            return {}
    
    @tool
    def get_startup_metrics_data(company_name: str) -> Dict[str, Any]:
        """
        Get comprehensive StartupMetrics data for a company.
        
        Args:
            company_name: The name of the company to search for
            
        Returns:
            Dictionary with StartupMetrics data
        """
        try:
            from langchain_core.pydantic_v1 import BaseModel, Field, create_model
            from langchain.output_parsers.pydantic import PydanticOutputParser
            from langchain_core.prompts import PromptTemplate
            from langchain_openai import ChatOpenAI
            from models.model import StartupMetrics
            
            # Create parser based on the StartupMetrics model
            parser = PydanticOutputParser(pydantic_object=StartupMetrics)
            
            # Get data from WebSearchUtils
            news_data = WebSearchUtils.search_news(company_name)
            company_info = WebSearchUtils.search_company_info(company_name)
            financial_data = WebSearchUtils.search_financial_data(company_name)
            category_data = WebSearchUtils.search_category_to_search_data(company_name)
            
            # Combine all data sources
            combined_data = {**company_info, **financial_data, **category_data}
            
            # Create a prompt that includes context and formatting instructions
            prompt = PromptTemplate(
                template="""Based on the following information about {company_name}, please extract 
                metrics that match the StartupMetrics model.
                
                Company Information: {company_info}
                News Data: {news_data}
                Financial Data: {financial_data}
                Additional Metrics: {category_data}
                
                {format_instructions}
                """,
                input_variables=["company_name", "company_info", "news_data", "financial_data", "category_data"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
            
            # Format the prompt with real data
            formatted_prompt = prompt.format(
                company_name=company_name,
                company_info=json.dumps(company_info),
                news_data=json.dumps(news_data),
                financial_data=json.dumps(financial_data),
                category_data=json.dumps(category_data)
            )
            
            # Get response from LLM
            llm = ChatOpenAI(temperature=0, model="gpt-4o")
            response = llm.invoke(formatted_prompt)
            
            # Parse the response into our model
            data = parser.parse(response.content)
            
            # Return as dictionary
            return data.model_dump()
        except Exception as e:
            print(f"Error getting StartupMetrics data: {str(e)}")
            return {}
    
    # For backward compatibility
    get_category_to_search_data = get_startup_metrics_data


class PDFAgentExecutor:
    """Executor for PDF extraction agent workflow."""
    
    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the PDF agent executor.
        
        Args:
            model_name: Name of the OpenAI model to use
        """
        # Set a timeout for API calls to avoid hanging
        self.timeout = 60
        
        try:
            self.llm = ChatOpenAI(temperature=0, model=model_name, request_timeout=self.timeout)
            
            # Create tools from PDFAgentTools methods
            self.tools = [
                PDFAgentTools.extract_company_info,
                PDFAgentTools.extract_financial_metrics,
                PDFAgentTools.extract_operational_metrics,
                PDFAgentTools.extract_strategic_and_market_metrics,
                PDFAgentTools.extract_founder_metrics,
                PDFAgentTools.enrich_with_web_data,
                PDFAgentTools.extract_social_profiles,
                PDFAgentTools.extract_linkedin_data,
                PDFAgentTools.search_news,
                PDFAgentTools.get_startup_metrics_data  # Use the new method instead of get_category_to_search_data
            ]
            
            # Add memory to maintain conversation context
            from langchain.memory import ConversationBufferMemory
            
            self.memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )
            
            # Create the system prompt
            system_prompt = """You are a specialized financial analyst agent that extracts structured data from startup pitch decks. 
            Your task is to extract and enrich startup data from PDF content. Follow these steps:
            
            1. Analyze the PDF content to extract company information
            2. Extract financial metrics
            3. Extract operational metrics
            4. Extract strategic and market metrics
            5. Extract founder metrics
            6. Enrich data with web sources if needed
            
            Make sure to use the appropriate tools for each task.
            """
            
            # Create the prompt template
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad")
            ])
            
            # Create the agent
            self.agent = create_openai_tools_agent(self.llm, self.tools, prompt)
            
            # Create the agent executor with memory
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True
            )
        except Exception as e:
            print(f"Error initializing PDFAgentExecutor: {str(e)}")
            # Will use fallback methods if initialization fails
    
    def extract_from_pdf_text(self, pdf_text: str, enable_web_enrichment: bool = True) -> Dict[str, Any]:
        """
        Extract data from PDF text using the agent workflow.
        
        Args:
            pdf_text: Text content of the PDF
            enable_web_enrichment: Whether to enrich with web data
            
        Returns:
            Dictionary with extracted and enriched data
        """
        try:
            # Check if the agent executor was properly initialized
            if not hasattr(self, 'agent_executor') or self.agent_executor is None:
                raise Exception("Agent executor not initialized properly")
                
            # Prepare the input for the agent
            input_data = {
                "input": f"Extract structured data from this pitch deck. If web enrichment is enabled ({enable_web_enrichment}), also search the web for additional information.\n\nPDF content:\n{pdf_text[:10000]}...",
                "chat_history": []
            }
            
            # Add timeout to avoid hanging
            start_time = time.time()
            
            # Execute the agent
            result = self.agent_executor.invoke(input_data)
            
            # Check if execution took too long
            if time.time() - start_time > self.timeout:
                print(f"Warning: Agent execution took longer than {self.timeout} seconds")
            
            # Process the result
            output = result["output"]
            
            # Look for JSON in the output
            start_idx = output.find("{")
            end_idx = output.rfind("}") + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = output[start_idx:end_idx]
                extracted_data = json.loads(json_str)
            else:
                # If no JSON is found, use basic extraction on the PDF text
                extracted_data = self._fallback_extraction(pdf_text)
            
            # Create a StartupMetrics instance
            metrics = StartupMetrics()
            
            # Populate metrics from extracted data
            for key, value in extracted_data.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
            
            # Try web enrichment if enabled
            if enable_web_enrichment and metrics.company_name:
                try:
                    company_name = metrics.company_name
                    from etl.util.model_util import enrich_startup_metrics_from_web
                    enriched_metrics = enrich_startup_metrics_from_web(company_name, metrics)
                    
                    # Return enriched data
                    return {
                        "metrics": enriched_metrics.model_dump(),
                        "main_category": enriched_metrics.model_dump(),
                        "search_category": enriched_metrics.model_dump()
                    }
                except Exception as e:
                    print(f"Web enrichment failed: {str(e)}")
                    # Continue with non-enriched data
            
            # Return non-enriched data
            return {
                "metrics": metrics.model_dump(),
                "main_category": metrics.model_dump(),
                "search_category": metrics.model_dump()
            }
            
        except requests.exceptions.RequestException as e:
            # Handle connection errors
            error_msg = f"Connection error: {str(e)}"
            print(error_msg)
            
            # Use fallback extraction
            fallback_data = self._fallback_extraction(pdf_text)
            metrics = StartupMetrics()
            
            # Set basic fields from fallback
            for key, value in fallback_data.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
            
            return {
                "error": error_msg,
                "metrics": metrics.model_dump(),
                "main_category": metrics.model_dump(),
                "search_category": metrics.model_dump()
            }
            
        except Exception as e:
            # Handle other errors
            error_msg = f"Error processing agent output: {str(e)}"
            print(error_msg)
            
            # Try fallback extraction as a last resort
            fallback_data = self._fallback_extraction(pdf_text)
            metrics = StartupMetrics()
            
            # Set basic fields from fallback
            for key, value in fallback_data.items():
                if hasattr(metrics, key):
                    setattr(metrics, key, value)
            
            return {
                "error": error_msg,
                "metrics": metrics.model_dump(),
                "main_category": metrics.model_dump(),
                "search_category": metrics.model_dump()
            }
    
    def _fallback_extraction(self, pdf_text: str) -> Dict[str, Any]:
        """
        Basic fallback extraction when agent-based extraction fails.
        Uses simple text analysis to extract key information.
        
        Args:
            pdf_text: The text content of the PDF
            
        Returns:
            Dictionary with basic extracted information
        """
        result = {}
        
        import re
        
        # Try multiple extraction methods to find the company name
        # Method 1: Look for domain names in URLs, which often contain company names
        url_patterns = [
            r'www\.([a-zA-Z0-9_-]+)\.com',
            r'https?://(?:www\.)?([a-zA-Z0-9_-]+)\.com',
            r'https?://(?:www\.)?([a-zA-Z0-9_-]+)\.org',
            r'https?://(?:www\.)?([a-zA-Z0-9_-]+)\.co',
            r'https?://(?:www\.)?([a-zA-Z0-9_-]+)\.io'
        ]
        
        # Try to extract a domain name, which might be the company name
        website_link = None
        domain_name = None
        
        for pattern in url_patterns:
            url_match = re.search(pattern, pdf_text)
            if url_match:
                domain = url_match.group(1).lower()
                website_link = url_match.group(0)
                domain_name = domain
                break
        
        # Method 2: Look for patterns like "Company:" or "About [Company]"
        company_pattern = re.compile(r'(?:Company|Organization|About)\s*:?\s*([A-Z][A-Za-z0-9\s]+(?:Inc\.?|LLC|Ltd\.?|Corporation|Corp\.?|Company|Co\.?)?)', re.IGNORECASE)
        company_match = company_pattern.search(pdf_text)
        
        if company_match:
            result["company_name"] = company_match.group(1).strip()
        elif domain_name:
            # If we found a domain but no explicit company name, use the domain
            result["company_name"] = domain_name.title()  # Capitalize properly
        else:
            # Method 3: Try to extract the first capitalized phrase as a potential company name
            lines = pdf_text.split('\n')
            for line in lines[:30]:  # Check first 30 lines for potential company names
                # Skip very short or empty lines
                if len(line.strip()) < 3:
                    continue
                    
                words = line.strip().split()
                if len(words) >= 2 and words[0][0].isupper():
                    # Take the first few words that might represent a company name
                    potential_name = ' '.join(words[:3])
                    
                    # Avoid common false positives 
                    false_positives = ['welcome', 'introduction', 'about', 'table of contents', 'content', 'section']
                    if not any(fp in potential_name.lower() for fp in false_positives):
                        result["company_name"] = potential_name
                        break
            
            # Method 4: Look for common company suffixes
            if "company_name" not in result:
                company_suffix_pattern = re.compile(r'([A-Z][A-Za-z0-9\s]+)\s+(Inc\.?|LLC|Ltd\.?|Corporation|Corp\.?|Company|Co\.?)', re.IGNORECASE)
                suffix_match = company_suffix_pattern.search(pdf_text)
                if suffix_match:
                    result["company_name"] = suffix_match.group(0).strip()
        
        # If all methods fail, use a generic placeholder
        if "company_name" not in result:
            result["company_name"] = "Unknown Company"
        
        # Record website URL if found
        if website_link:
            result["website_link"] = website_link
        
        # Extract a summary from the first 1000 characters
        summary = pdf_text[:1000]
        result["pitch_deck_summary"] = summary
        
        # Try to extract founders and team information
        founders = []
        founder_info = ""
        
        # Find team/founder section
        team_sections = re.split(r'(?i)(?:team|founders|management|leadership|executives|about\s+us)(?:\s+section)?[\s\:\-]+', pdf_text)
        if len(team_sections) > 1:
            team_section = team_sections[1].split('\n\n')[0]  # Take first paragraph after team heading
            
            # Look for common patterns in team sections
            # Pattern 1: Full name followed by title with CEO/Founder/Co-founder
            founder_pattern = re.compile(r'([A-Z][a-z]+\s+[A-Z][a-z]+)[\s\,\-]+(?:CEO|Chief\s+Executive\s+Officer|Founder|Co-founder|Cofounder)', re.IGNORECASE)
            founder_matches = founder_pattern.finditer(team_section)
            for match in founder_matches:
                founders.append(match.group(1))
            
            # Pattern 2: Name followed by email that includes domain
            email_pattern = re.compile(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)[\s\,\.\:\-]*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', re.IGNORECASE)
            email_matches = email_pattern.finditer(pdf_text)
            
            for match in email_matches:
                full_name = match.group(1).strip()
                email = match.group(2)
                
                # Only add if we don't already have this founder
                if full_name not in founders:
                    founders.append(full_name)
                    
                    # Extract domain part from email as additional company confirmation
                    email_domain = email.split('@')[-1].split('.')[0]
                    if domain_name is None and email_domain not in ['gmail', 'hotmail', 'yahoo', 'outlook']:
                        domain_name = email_domain
        
        # If we found any founders, format them and search for LinkedIn profile
        if founders:
            founder_info = ", ".join(founders)
            result["founders"] = founder_info
            
            # If we don't already have a LinkedIn link, try to create one for the first founder
            if "linkedin_profile_ceo" not in result and len(founders) > 0:
                try:
                    # Create a LinkedIn URL from founder name
                    ceo_name = founders[0]
                    name_parts = ceo_name.strip().split()
                    
                    if len(name_parts) >= 2:
                        first_name = name_parts[0].lower()
                        last_name = '-'.join([part.lower() for part in name_parts[1:]])
                        company_slug = result.get("company_name", "").lower().replace(" ", "-")
                        
                        # Remove special characters
                        company_slug = re.sub(r'[^a-z0-9\-]', '', company_slug)
                        first_name = re.sub(r'[^a-z0-9\-]', '', first_name)
                        last_name = re.sub(r'[^a-z0-9\-]', '', last_name)
                        
                        # Try to search for the LinkedIn profile using the WebSearchUtils
                        try:
                            linkedin_data = WebSearchUtils.search_linkedin(
                                first_name=first_name,
                                last_name=last_name,
                                company_name=result.get("company_name", "")
                            )
                            
                            # If the search returned data, we've found a profile
                            if linkedin_data and isinstance(linkedin_data, dict) and len(linkedin_data) > 0:
                                # The API doesn't directly return the URL, so construct a likely URL
                                result["linkedin_profile_ceo"] = f"https://linkedin.com/in/{first_name}-{last_name}"
                        except Exception as e:
                            print(f"Error searching LinkedIn: {str(e)}")
                            # Fallback to a constructed URL even if the search failed
                            result["linkedin_profile_ceo"] = f"https://linkedin.com/in/{first_name}-{last_name}"
                except Exception as e:
                    print(f"Error creating LinkedIn URL: {str(e)}")
        
        # Try to extract additional contact information
        email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        email_match = email_pattern.search(pdf_text)
        if email_match:
            result["contact_email"] = email_match.group(0)
        
        # Try to extract founding year
        year_pattern = re.compile(r'(?:founded|established|since|est\.?)\s+in\s+(\d{4})', re.IGNORECASE)
        year_match = year_pattern.search(pdf_text)
        if year_match:
            try:
                result["year_of_founding"] = int(year_match.group(1))
            except ValueError:
                pass
        
        # Try to extract location/headquarters
        location_pattern = re.compile(r'(?:headquartered|based|location|address|hq)(?:\s+in)?\s+([A-Za-z\s,]+(?:USA|US|United States|Canada|UK|Australia|[A-Z]{2}))', re.IGNORECASE)
        location_match = location_pattern.search(pdf_text)
        if location_match:
            result["location_of_headquarters"] = location_match.group(1).strip()
        
        return result
    
    def _format_to_category(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the extracted data to match the Category model structure.
        
        Args:
            data: Dictionary with extracted data
            
        Returns:
            Dictionary formatted according to the Category model
        """
        # Create an empty Category instance
        category = Category()
        
        # Process company info
        if "company_info" in data:
            company_info = CompanyInfo(**data["company_info"])
            category.company_info = company_info
        elif any(key.startswith("company_") for key in data.keys()):
            # If company info is in the root level, gather it
            company_info_fields = {}
            for key in list(data.keys()):
                if key in CompanyInfo.model_fields:
                    company_info_fields[key] = data.pop(key)
            
            category.company_info = CompanyInfo(**company_info_fields)
        
        # Transfer rest of the fields to the Category model
        for key, value in data.items():
            if hasattr(category, key):
                setattr(category, key, value)
        
        return category.model_dump()