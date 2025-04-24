from pydantic.v1 import BaseModel


class NewsModel(BaseModel):
    title: str
    description: str
    tone: str
    keywords: list[str]
    summary: str
    url: str
    source: str