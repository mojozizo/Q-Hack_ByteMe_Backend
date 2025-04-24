import os
import json
from dotenv import load_dotenv
from etl.agent.linkedin_agent import LinkedInAgent

# Load environment variables
load_dotenv()

# Check if OpenAI API key is set
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("Error: OPENAI_API_KEY environment variable is not set.")
    print("Please set it in your .env file or export it in your terminal.")
    exit(1)
    
print(f"Using OpenAI API key: {openai_api_key[:5]}...{openai_api_key[-4:] if len(openai_api_key) > 8 else ''}")

try:
    # Initialize the agent
    linkedin_agent = LinkedInAgent()
    
    # Real LinkedIn profile URL - use a public profile for testing
    profile_url = 'https://www.linkedin.com/in/satyanadella/'
    
    print(f"Testing LinkedIn agent with profile: {profile_url}")
    result = linkedin_agent._run(profile_url)
    
    # Try to parse the result as JSON for prettier display
    try:
        parsed_result = json.loads(result)
        print(f'Extracted LinkedIn data:\n{json.dumps(parsed_result, indent=2)}')
    except:
        print(f'Extracted LinkedIn data: {result}')
        
except Exception as e:
    print(f"Error occurred while running LinkedIn agent: {str(e)}")
