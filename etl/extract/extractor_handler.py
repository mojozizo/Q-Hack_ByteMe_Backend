from etl.extract.abstract_extracter import AbstractExtracter


class ExtractorHandler:
    """
    This class is responsible for handling the extraction of data from various sources.
    It uses the appropriate extractor based on the source type.
    """
    @staticmethod
    def get_extractor(source_type: str) -> AbstractExtracter:
        """
        Returns the appropriate extractor based on the source type.
        """
        if source_type == "pdf":
            from etl.extract.pdf_extracter import PDFExtracter
            return PDFExtracter()
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
