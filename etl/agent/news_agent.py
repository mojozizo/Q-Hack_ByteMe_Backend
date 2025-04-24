import json
import os
import re
import traceback

from langchain_openai import ChatOpenAI

from etl.transform.parsers.news_api_parser import NewsAPIClientParser
from models.news_model import NewsModel


class NewsAgent:
    """Agent for news data extraction."""

    def __init__(self):
        """Initialize the News agent with OpenAI client."""
        self.name = "News Agent"
        # Initialize the LLM
        self.llm = ChatOpenAI(
            temperature=0.3,
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def _run(self, input_query: str) -> str:
        """
        Run the agent with the given news query.

        Args:
            input_query: News search query

        Returns:
            Structured JSON data of the news articles
        """
        # Use the NewsParser to parse the input query
        parser = NewsAPIClientParser(query=input_query)
        parsed_data = parser.parse()

        # Process the parsed data
        processed_data = self.process_parsed_data(parsed_data)

        return processed_data

    def process_parsed_data(self, parsed_data: dict) -> str:
        """Process parsed news data into a structured model."""
        # Create a prompt for the LLM
        prompt = f"""
        Extract the following information from this news data:
        - Title
        - Description
        - Tone of the article
        - Keywords
        - Summary
        - URL
        - Source
        - Author
        - Published date

        Return ONLY a valid JSON object with these fields: title, description, tone, keywords (as array), summary, url, source, author, published_date

        News data: {json.dumps(parsed_data)}
        """

        response = self.llm.invoke(prompt)
        content = response.content

        extracted_data = {}
        debug_info = {"raw_response": content}

        try:
            extracted_data = json.loads(content)
            debug_info["parsing_method"] = "direct_json_parse"
        except json.JSONDecodeError:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', content)
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                extracted_data = json.loads(json_str)
                debug_info["parsing_method"] = "regex_extraction"
            else:
                raise ValueError("Could not parse JSON from response")

        title = str(extracted_data.get("title", "") or "")
        description = str(extracted_data.get("description", "") or "")
        tone = str(extracted_data.get("tone", "") or "")
        summary = str(extracted_data.get("summary", "") or "")
        url = str(extracted_data.get("url", "") or "")
        source = str(extracted_data.get("source", "") or "")
        author = str(extracted_data.get("author", "") or "")
        published_date = str(extracted_data.get("published_date", "") or "")
        keywords = extracted_data.get("keywords", [])

        if not isinstance(keywords, list):
            keywords = []

        news_model = NewsModel(
            title=title,
            description=description,
            tone=tone,
            keywords=keywords,
            summary=summary,
            url=url,
            source=source
        )

        return news_model
