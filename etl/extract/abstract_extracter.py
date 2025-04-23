import abc

from typing import Dict, List, Optional

from fastapi import Path


class AbstractExtracter(abc.ABC):
    """
    Abstract base class for extracters.
    """

    def __init__(self):
        """
        Initialize the extractor.
        """
        pass

    @abc.abstractmethod
    def extract(self, file: Path, query: str) -> List[Dict]:
        """
        Extract data from a source.

        Returns:
            List[Dict]: A list of dictionaries containing the extracted data.
        """
        pass