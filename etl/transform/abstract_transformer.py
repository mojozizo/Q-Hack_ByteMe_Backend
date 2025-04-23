import abc


class AbstractTransformer(abc.ABC):
    """
    Abstract base class for transformers.
    """

    def __init__(self):
        """
        Initialize the transformer.
        """
        pass

    @abc.abstractmethod
    def transform(self, data: dict) -> dict:
        """
        Transform the data.

        Args:
            data (dict): The data to transform.

        Returns:
            dict: The transformed data.
        """
        pass