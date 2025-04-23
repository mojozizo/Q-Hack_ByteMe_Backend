import json
import os

from dotenv import load_dotenv
from langchain.agents import Agent
from langchain_openai import ChatOpenAI

from etl.transform.parsers.linkedin_parser import LinkedInParser
from models.linkedin_owner_model import LinkedInOwnerModel

# Load environment variables before initializing any clients
load_dotenv()

class LinkedInAgent(Agent):
    """Agent for LinkedIn."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "LinkedIn Agent"
        self.description = "An agent that processes data from LinkedIn that was parsed by LinkedInParser."
        self.tools = [
            {
                "name": "LinkedIn Parser",
                "description": "Parses LinkedIn data.",
            }
        ]
        # Initialize OpenAI client with API key
        self.openai_client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _run(self, input: str) -> str:
        """
        Run the agent with the given input.
        """
        # Use the LinkedInParser to parse the input
        parser = LinkedInParser()
        parsed_data = parser.parse_by_url(input)

        # Process the parsed data
        processed_data = self.process_parsed_data(parsed_data)

        return processed_data

    def process_parsed_data(self, parsed_data: dict) -> str:
        """
        Process the parsed data and converts it to the LinkedInOwnerModel based on the schema.
        """
        try:
            # Create a prompt for OpenAI to extract structured information
            prompt = f"""
            Extract the following information from this LinkedIn profile data:
            - Full name
            - Professional title/position
            - Location
            - Professional summary
            - List of skills
            - Current company name

            LinkedIn data: {json.dumps(parsed_data)}

            Return ONLY a valid JSON object with these fields: name, title, location, summary, skills (as array), current_company
            """

            # Call OpenAI API using the current API pattern
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a data extraction assistant. Extract structured data from LinkedIn profiles."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            # Parse the response
            extracted_data = json.loads(response.choices[0].message.content.strip())

            # Create and validate with the model
            linkedin_model = LinkedInOwnerModel(
                name=extracted_data.get("name", ""),
                title=extracted_data.get("title"),
                location=extracted_data.get("location"),
                summary=extracted_data.get("summary"),
                skills=extracted_data.get("skills", []),
                current_company=extracted_data.get("current_company")
            )

            return linkedin_model.model_dump_json()
        except Exception as e:
            return f"Error processing LinkedIn data: {str(e)}"