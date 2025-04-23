from typing import Optional

from pydantic import BaseModel


class LinkedInOwnerModel(BaseModel):
    name: str
    title: Optional[str]
    location: Optional[str]
    summary: Optional[str]
    skills: Optional[list]
    current_company: Optional[str]