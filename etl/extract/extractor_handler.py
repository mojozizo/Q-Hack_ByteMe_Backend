from etl.extract.abstract_extracter import AbstractExtracter


class ExtractorHandler:
    """
    This class is responsible for handling the extraction of data from various sources.
    It uses the appropriate extractor based on the source type.
    """
    @staticmethod
    def get_extractor(source_type: str, use_agent_workflow: bool = False) -> AbstractExtracter:
        """
        Returns the appropriate extractor based on the source type.
        
        Args:
            source_type: The type of source to extract data from (e.g., "pdf")
            use_agent_workflow: Whether to use the LangChain agent workflow (default: False)
            
        Returns:
            An instance of the appropriate extractor
        """
        if source_type == "pdf":
            from etl.extract.pdf_extracter import PDFExtracter
            return PDFExtracter(use_agent_workflow=use_agent_workflow)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
