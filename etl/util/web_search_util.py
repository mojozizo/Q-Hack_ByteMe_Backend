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
    def search_cik_by_name(company_name: str) -> Optional[str]:
        """
        Search for a company's CIK number using OpenAI's web browsing capability.

        Args:
            company_name: Company name to search for

        Returns:
            CIK number as a string (padded to 10 digits) if found, None otherwise
        """
        prompt = f"""
        Find the SEC Central Index Key (CIK) for the company "{company_name}".

        Search the SEC Edgar database or other reliable sources to find this information.
        The CIK is a 10-digit number (may include leading zeros) that the SEC uses to identify companies.

        Format your response as a JSON object with the key "cik" containing the CIK number as a string.
        If you find a CIK with fewer than 10 digits, pad it with leading zeros to make it 10 digits.
        If you cannot find the CIK with high confidence, set the value to null.

        Example response:
        {{
          "cik": "0000320193"
        }}
        """

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",
                     "content": "You are a financial research assistant that finds accurate company CIK numbers."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            cik = result.get("cik")

            if not cik or cik == "null":
                return None

            return str(cik).zfill(10)
        except Exception as e:
            print(f"CIK lookup failed: {str(e)}")
            return None

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
    
    @staticmethod
    def search_category_to_search_data(company_name: str) -> Dict[str, Any]:
        """
        Search for additional startup metrics defined in the CategoryToSearch model using a combination
        of web search, BrightData, and News API.
        
        Args:
            company_name: Company name to search for
            
        Returns:
            Dictionary containing CategoryToSearch metrics
        """
        # First collect data from multiple sources
        web_data = {}
        news_data = {}
        linkedin_data = {}
        
        # Get news data
        try:
            news_data = WebSearchUtils.search_news(company_name)
        except Exception as e:
            print(f"News API search failed: {str(e)}")
        
        # Try to get LinkedIn company page
        try:
            # First try to get the company website
            company_info = WebSearchUtils.search_company_info(company_name)
            if "website_link" in company_info:
                # Extract social profiles to find LinkedIn
                social_data = WebSearchUtils.extract_social_profiles(company_info["website_link"])
                if "linkedin" in social_data and social_data["linkedin"]:
                    # Use LinkedIn profile URL to get data
                    linkedin_data = WebSearchUtils.search_linkedin(profile_url=social_data["linkedin"])
        except Exception as e:
            print(f"LinkedIn company search failed: {str(e)}")
        
        # Create prompt for GPT-4o with all collected data as context
        advanced_metrics_prompt = f"""
        I need detailed metrics for the company {company_name}. 
        
        Here's what I've collected about the company so far:
        
        News Data: {json.dumps(news_data) if news_data else "No news data available"}
        
        LinkedIn Data: {json.dumps(linkedin_data) if linkedin_data else "No LinkedIn data available"}
        
        Based on this information and your knowledge, please extract or estimate the following metrics:
        
        1. Churn Rate (percentage of users lost monthly)
        2. Net Revenue Retention (NRR)
        3. Customer Payback Period (in months)
        4. DAU/MAU Ratio (as percentage)
        5. Product Stickiness (scale 1-5)
        6. Burn Multiple (cash burn / net new revenue)
        7. Time to Value (TTV) in days
        8. Revenue per FTE
        9. Valuation / ARR Multiple
        10. Top-3 Revenue Share (percentage from top 3 customers)
        11. Market Coverage (Revenue / SAM) as percentage
        12. Employee Count
        13. Business Model Scalability (scale 1-5)
        14. Hiring Plan Alignment (scale 1-5)
        15. Any Regulatory Risks (true/false)
        16. Any Trend Risks (true/false)
        17. Any Litigation or IP Disputes (true/false)
        18. Is Founder Sanction Free (true/false)
        19. Is Company Sanction Free (true/false)
        
        Format your response as a JSON object with these exact field names:
        - churn_rate (integer percentage)
        - net_revenue_retention (integer percentage)
        - customer_payback_period (integer months)
        - dau_mau_ratio (integer percentage)
        - product_stickiness (integer 1-5)
        - burn_multiple (integer)
        - time_to_value (integer days)
        - revenue_per_fte (integer USD)
        - valuation_arr_multiple (integer)
        - top_3_revenue_share (integer percentage)
        - market_coverage (integer percentage)
        - employee_count (integer)
        - business_model_scalability (integer 1-5)
        - hiring_plan_alignment (integer 1-5)
        - regulatory_risks (boolean)
        - trend_risks (boolean)
        - litigation_ip_disputes (boolean)
        - founder_sanction_free (boolean)
        - company_sanction_free (boolean)
        
        For numeric values, convert to integers (round as needed).
        Only include fields where you have reasonable confidence. Omit fields where you're highly uncertain.
        """
        
        try:
            # Use GPT-4o to generate the metrics
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a startup analyst with expertise in SaaS and tech metrics. You have access to various data sources and can make reasonable estimations based on available information."},
                    {"role": "user", "content": advanced_metrics_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            # Parse and return the result
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            print(f"CategoryToSearch metrics search failed: {str(e)}")
            return {}