from newsapi import NewsApiClient
from etl.transform.parsers.abstract_parser import AbstractParser
from etl.util.token_util import get_newsapi_token

class NewsAPIClientParser(AbstractParser):
    """
    Parser for NewsAPI.org using the unofficial Python client library.
    """

    def __init__(
        self,
        query: str,
        language: str = "en",
        sources: str = None,
        sort_by: str = "relevancy",
        page_size: int = 10,
        page: int = 1,
        from_date: str = None,
        to_date: str = None
    ):
        super().__init__()
        self.query = query
        self.language = language
        self.sources = sources
        self.sort_by = sort_by
        self.page_size = page_size
        self.page = page
        self.from_date = from_date
        self.to_date = to_date

        self.client = NewsApiClient(api_key=get_newsapi_token())

    def parse(self) -> dict:
        """
        Query the NewsAPI endpoint via the Python client.
        """
        params = {
            "q": self.query,
            "language": self.language,
            "sort_by": self.sort_by,
            "page_size": self.page_size,
            "page": self.page,
        }
        if self.sources:
            params["sources"] = self.sources
        if self.from_date:
            params["from_param"] = self.from_date
        if self.to_date:
            params["to"] = self.to_date

        resp = self.client.get_everything(**params)

        articles = []
        for art in resp.get("articles", []):
            articles.append({
                "source": art.get("source", {}).get("name"),
                "author": art.get("author"),
                "title": art.get("title"),
                "description": art.get("description"),
                "url": art.get("url"),
                "publishedAt": art.get("publishedAt"),
                "content": art.get("content"),
            })

        return {
            "query": self.query,
            "totalResults": resp.get("totalResults"),
            "articles": articles
        }
