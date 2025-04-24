#!/usr/bin/env python3
"""
Test script for the Financial Agent.

This script tests the Financial Agent's ability to extract financial information
from SEC EDGAR for a given company.

Usage:
    python test_financial_agent.py [company_name]
    
    If no company name is provided, the script will use default test companies.
"""

import json
import sys
import os
from pprint import pprint

# Add the parent directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from etl.agent.financial_agent import FinancialAgent
from models.financial_model import FinancialModel

def test_financial_agent(company_name: str) -> None:
    """
    Test the Financial Agent with a given company name.
    
    Args:
        company_name: Name of the company to look up
    """
    print(f"\n{'=' * 50}")
    print(f"Testing Financial Agent with company: {company_name}")
    print(f"{'=' * 50}")
    
    # Initialize the financial agent
    agent = FinancialAgent()
    
    try:
        # Run the agent with the company name
        print("Fetching financial data...")
        result_json = agent._run(company_name)
        
        # Parse the JSON result
        if result_json:
            try:
                # Try to parse the JSON string into a Python dictionary
                result = json.loads(result_json)
                
                # Print the results in a readable format
                print("\nFinancial Information:")
                print(f"Company Name: {result.get('company_name', 'N/A')}")
                print(f"CIK Number: {result.get('cik', 'N/A')}")
                print(f"Industry (SIC): {result.get('sic_description', 'N/A')}")
                print(f"Revenue: {result.get('revenue', 'N/A')}")
                print(f"Net Income: {result.get('net_income', 'N/A')}")
                print(f"Total Assets: {result.get('total_assets', 'N/A')}")
                print(f"Total Liabilities: {result.get('total_liabilities', 'N/A')}")
                print(f"Fiscal Year: {result.get('fiscal_year', 'N/A')}")
                
                # Print recent filings
                if "recent_filings" in result and result["recent_filings"]:
                    print("\nRecent Filings:")
                    for filing in result["recent_filings"]:
                        print(f"- {filing.get('form', 'N/A')} ({filing.get('filingDate', 'N/A')}): {filing.get('description', 'N/A')}")
                
                # Print financial summary
                if "financial_summary" in result and result["financial_summary"]:
                    print("\nFinancial Summary:")
                    print(result["financial_summary"])
                
                # Check for errors
                if "error" in result:
                    print(f"\nError: {result['error']}")
                
                # Check if technical details are available
                if "technical" in result and isinstance(result["technical"], dict):
                    if "error" in result["technical"]:
                        print(f"\nTechnical Error: {result['technical']['error']}")
                
                print("\nRaw JSON Result:")
                print(json.dumps(result, indent=2))
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print("Raw result:")
                print(result_json)
        else:
            print("No result returned from the Financial Agent")
    
    except Exception as e:
        print(f"Error testing Financial Agent: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point for the script."""
    # Get company name from command line arguments or use defaults
    if len(sys.argv) > 1:
        company_name = sys.argv[1]
        test_financial_agent(company_name)
    else:
        # Test with some default companies
        test_companies = [
            "Apple",
            "Microsoft",
            "Tesla",
            "Amazon"
        ]
        
        for company in test_companies:
            test_financial_agent(company)
            print("\n")  # Add spacing between tests

if __name__ == "__main__":
    main()