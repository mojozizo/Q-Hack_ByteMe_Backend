from etl.extract.abstract_extracter import AbstractExtracter
from typing import Dict, Any, Optional


class ExtractorHandler:
    """
    This class is responsible for handling the extraction of data from various sources.
    It uses the appropriate extractor based on the source type.
    """
    @staticmethod
    def get_extractor(source_type: str, use_agent_workflow: bool = False, use_modular_workflow: bool = False) -> AbstractExtracter:
        """
        Returns the appropriate extractor based on the source type.
        
        Args:
            source_type: The type of source to extract data from (e.g., "pdf")
            use_agent_workflow: Whether to use the LangChain agent workflow (default: False)
            use_modular_workflow: Whether to use the new modular workflow with retrievers (default: False)
            
        Returns:
            An instance of the appropriate extractor
        """
        if source_type == "pdf":
            # Determine which PDF extractor to use based on the requested workflow
            if use_modular_workflow:
                from etl.extract.modular_extracter import ModularExtractor
                return ModularExtractor()
            elif use_agent_workflow:
                from etl.extract.pdf_extracter import PDFExtracter
                return PDFExtracter(use_agent_workflow=True)
            else:
                # Use WebSearch-enhanced PDF extractor 
                from etl.extract.pdf_web_search_extractor import PDFWebSearchExtractor
                return PDFWebSearchExtractor()
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
