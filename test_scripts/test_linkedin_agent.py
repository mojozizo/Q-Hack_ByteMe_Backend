#!/usr/bin/env python3
import sys
import json
import os
from pathlib import Path

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

from etl.agent.linkedin_agent import LinkedInAgent

def main():
    """
    Test the LinkedIn Agent with a LinkedIn profile URL.
    
    Usage:
        python test_linkedin_agent.py "LinkedIn Profile URL"
    
    Example:
        python test_linkedin_agent.py "https://www.linkedin.com/in/satyanadella/"
    """
    if len(sys.argv) < 2:
        print("Please provide a LinkedIn profile URL.")
        print("Usage: python test_linkedin_agent.py \"LinkedIn Profile URL\"")
        sys.exit(1)
    
    # Get the LinkedIn profile URL from command line arguments
    linkedin_url = sys.argv[1]
    print(f"Processing LinkedIn profile: {linkedin_url}")
    
    # Initialize the LinkedIn Agent
    linkedin_agent = LinkedInAgent()
    
    try:
        # Run the agent with the LinkedIn profile URL
        result = linkedin_agent._run(linkedin_url)
        
        # Pretty print the result
        print("\nLinkedIn Agent Result:")
        print("======================")
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