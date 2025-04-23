from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import re

from etl.transform.parsers.linkedin_parser import LinkedInParser
from models.linkedin_owner_model import LinkedInOwnerModel

# Load environment variables
load_dotenv()

class LinkedInAgent:
    """Agent for LinkedIn data extraction."""

    def __init__(self):
        """Initialize the LinkedIn agent with OpenAI client."""
        self.name = "LinkedIn Agent"
        # Initialize the LLM
        self.llm = ChatOpenAI(
            temperature=0.3,
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.parser = LinkedInParser()

    def _run(self, input_url: str) -> str:
        """
        Run the agent with the given LinkedIn URL.

        Args:
            input_url: LinkedIn profile URL

        Returns:
            Structured JSON data of the LinkedIn profile
        """
        # Use the LinkedInParser to parse the input URL
        parsed_data = self.parser.parse_by_url(input_url)

        # Process the parsed data
        processed_data = self.process_parsed_data(parsed_data)

        return processed_data

    def process_parsed_data(self, parsed_data: dict) -> str:
        """Process parsed LinkedIn data into a structured model."""
        try:
            # Create a prompt for the LLM
            prompt = f"""
            Extract the following information from this LinkedIn profile data:
            - Full name
            - Professional title/position
            - Location
            - Professional summary - not just take from linkedin, but also add your own summary based on the data. 
            You should act as a professional recruiter. 
            Dont just copy the summary from LinkedIn and trust everything that is said there. 
            analyze working experience, skills and education and create a summary based on that.
            - Full list of skills
            - Current company name

            LinkedIn data: {json.dumps(parsed_data)}

            Return ONLY a valid JSON object with these fields: name, title, location, summary, skills (as array), current_company
            """
            response = self.llm.invoke(prompt)
            content = response.content

            try:
                extracted_data = json.loads(content)
            except json.JSONDecodeError:
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', content)
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                    extracted_data = json.loads(json_str)
                else:
                    raise ValueError("Could not parse JSON from response")

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