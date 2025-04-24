#!/usr/bin/env python3
import sys
import json
import os
from pathlib import Path

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from etl.agent.news_agent import NewsAgent

def main():
    """
    Test the News Agent with a company name to search for news articles.
    
    Usage:
        python test_news_agent.py "Company Name"
    
    Example:
        python test_news_agent.py "Microsoft"
    """
    if len(sys.argv) < 2:
        print("Please provide a company name to search for news.")
        print("Usage: python test_news_agent.py \"Company Name\"")
        sys.exit(1)
    
    # Get the company name from command line arguments
    company_name = sys.argv[1]
    print(f"Searching for news about: {company_name}")
    
    # Initialize the News Agent
    news_agent = NewsAgent()
    
    try:
        # Run the agent with the company name
        result = news_agent._run(company_name)
        
        # Pretty print the result
        print("\nNews Agent Result:")
        print("==================")
        if isinstance(result, str):
            try:
                # Try to parse as JSON for pretty printing
                parsed_result = json.loads(result)
                print(json.dumps(parsed_result, indent=2))
            except json.JSONDecodeError:
                # If not JSON, print as is
                print(result)
        else:
            # If it's already an object, convert to JSON
            print(json.dumps(result, indent=2, default=str))
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()