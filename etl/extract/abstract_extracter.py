import abc

from typing import Dict, List, Optional

class AbstractExtracter(abc.ABC):
    """
    Abstract base class for extracters.
    """

    def __init__(self, config: Dict):
        self.config = config

    @abc.abstractmethod
    def extract(self) -> List[Dict]:
        """
        Extract data from a source.

        Returns:
            List[Dict]: A list of dictionaries containing the extracted data.
        """
        pass