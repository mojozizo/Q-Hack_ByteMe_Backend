import json
import os
from typing import Dict, Optional, List, Any

from openai import OpenAI
from etl.transform.parsers.linkedin_parser import LinkedInParser
from etl.transform.parsers.news_api_parser import NewsAPIClientParser

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class WebSearchUtils:
    """
    Utility class for web search functionality to find company and financial data
    that may be missing from pitch decks.
    """
    
    @staticmethod
    def search_linkedin(
        first_name: Optional[str] = None, 
        last_name: Optional[str] = None, 
        company_name: Optional[str] = None,
        profile_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search LinkedIn for information about a company or person.
        
        Args:
            first_name: First name of the person to search for
            last_name: Last name of the person to search for
            company_name: Company name to search for
            profile_url: Direct LinkedIn profile URL
            
        Returns:
            Dictionary containing LinkedIn data
        """
        try:
            linkedin_parser = LinkedInParser()
            
            # Try by URL if provided
            if profile_url:
                try:
                    data = linkedin_parser.parse_by_url(profile_url=profile_url)
                    return data
                except Exception as e:
                    print(f"LinkedIn search by URL failed: {str(e)}")
            
            # Try by name if provided
            if first_name and last_name:
                try:
                    data = linkedin_parser.parse_by_name(
                        first_name=first_name,
                        last_name=last_name,
                        company_name=company_name or ""
                    )
                    return data
                except Exception as e:
                    print(f"LinkedIn search by name failed: {str(e)}")
            
            return {}
        except Exception as e:
            print(f"LinkedIn search failed: {str(e)}")
            return {}
    
    @staticmethod
    def search_news(company_name: str, limit: int = 10) -> Dict[str, Any]:
        """
        Search news articles for information about a company.
        
        Args:
            company_name: Company name to search for
            limit: Maximum number of results to return
            
        Returns:
            Dictionary containing news data
        """
        try:
            news_parser = NewsAPIClientParser(
                query=company_name,
                page_size=limit
            )
            data = news_parser.parse()
            return data
        except Exception as e:
            print(f"News search failed: {str(e)}")
            return {}
    
    @staticmethod
    def search_financial_data(company_name: str) -> Dict[str, Any]:
        """
        Search for financial data about a company using OpenAI to search the web.
        
        Args:
            company_name: Company name to search for
            
        Returns:
            Dictionary containing financial data
        """
        prompt = f"""
        Search the web for financial information about {company_name}. 
        Specifically look for:
        1. Annual Recurring Revenue (ARR)
        2. Monthly Recurring Revenue (MRR)
        3. Customer Acquisition Cost (CAC)
        4. Customer Lifetime Value (CLTV)
        5. Gross Margin
        6. Revenue Growth Rate (YoY and MoM)
        7. Monthly Active Users (MAU)
        8. Sales Cycle Length
        9. Burn Rate
        10. Runway
        
        Format the response as JSON with keys matching the financial metrics in the following list:
        - annual_recurring_revenue (integer)
        - monthly_recurring_revenue (integer)
        - customer_acquisition_cost (integer)
        - customer_lifetime_value (integer)
        - cltv_cac_ratio (integer)
        - gross_margin (integer percentage)
        - revenue_growth_rate_yoy (integer percentage)
        - revenue_growth_rate_mom (integer percentage)
        - monthly_active_users (integer)
        - sales_cycle_length (integer days)
        - burn_rate (integer USD)
        - runway (integer months)
        
        Only include facts you find from reliable sources, not estimates or guesses.
        Convert all values to integers.
        """
        
        try:
            # Create a completion using OpenAI with browsing capabilities
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial research assistant that searches the web for accurate company information."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"Financial data search failed: {str(e)}")
            return {}
    
    @staticmethod
    def extract_social_profiles(website_url: str) -> Dict[str, str]:
        """
        Use OpenAI to find social media profiles from a company website.
        
        Args:
            website_url: URL of company website
            
        Returns:
            Dictionary mapping social platform names to profile URLs
        """
        prompt = f"""
        Visit the website {website_url} and extract the following information:
        1. LinkedIn company page URL
        2. Twitter/X profile URL 
        3. Facebook page URL
        4. CEO/Founder LinkedIn profile URL
        
        Format the response as a JSON object with keys: "linkedin", "twitter", "facebook", "ceo_linkedin".
        Only include URLs you actually find, don't make assumptions.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a web scraping assistant that finds social media links."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"Social profile extraction failed: {str(e)}")
            return {}
    
    @staticmethod
    def search_company_info(company_name: str) -> Dict[str, Any]:
        """
        Search for basic company information using OpenAI to search the web.
        
        Args:
            company_name: Company name to search for
            
        Returns:
            Dictionary containing company information
        """
        prompt = f"""
        Search the web for basic information about {company_name}. Find:
        1. Year of founding
        2. Location of headquarters 
        3. Industry
        4. Business model (B2B, B2C, etc.)
        5. Number of employees
        6. Website URL
        7. Short company description/pitch
        
        Format the response as JSON with these keys:
        - year_of_founding (integer)
        - location_of_headquarters (string)
        - industry (string)
        - business_model (string)
        - employees (string like "10-50")
        - website_link (string)
        - one_sentence_pitch (string)
        
        Only include facts you find from reliable sources, not estimates or guesses.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a company research assistant that searches the web for accurate company information."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"Company info search failed: {str(e)}")
            return {}