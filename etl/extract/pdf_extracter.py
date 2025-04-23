from etl.extract.abstract_extracter import AbstractExtracter


class PDFExtracter(AbstractExtracter):
    """
    Extracter for PDF files.
    """

    def __init__(self, config: dict):
        super().__init__(config)

    def extract(self) -> str:
        """
        Extract text from the PDF file.

        Returns:
            str: Extracted text from the PDF.
        """
        pass
