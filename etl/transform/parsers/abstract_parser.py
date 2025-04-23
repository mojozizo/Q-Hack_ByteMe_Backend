import abc


class AbstractParser(abc.ABC):
    """
    Abstract base class for parsers.
    """

    def __init__(self):
        """
        Initialize the parser.
        """
        pass

    @abc.abstractmethod
    def parse(self) -> dict:
        """
        Parse the data.

        Args:
            data (str): The data to parse.

        Returns:
            dict: The parsed data.
        """
        pass