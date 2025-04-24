import os
import shutil
import json
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import UploadFile
from etl.extract.abstract_extracter import AbstractExtracter
from etl.util.file_util import create_or_get_upload_folder
from etl.extract.simple_pdf_extractor import SimplePDFExtractor
from etl.agent.web_search_agent import WebSearchAgent


class PDFWebSearchExtractor(AbstractExtracter):
    """
    An extractor that combines PDF extraction with web search.
    First extracts data from a PDF, then uses web search to fill in missing values.
    """

    def __init__(self, model_name: str = "gpt-4o"):
        """
        Initialize the PDF web search extractor.
        
        Args:
            model_name: The OpenAI model to use for analysis
        """
        super().__init__()
        self.pdf_extractor = SimplePDFExtractor(model_name=model_name)
        self.web_search_agent = WebSearchAgent(model_name=model_name)

    def extract(self, file: UploadFile, query: str = None) -> str:
        """
        Extract data from a PDF and enhance it with web search results.
        
        Args:
            file: The uploaded PDF file
            query: Custom extraction prompt (optional)
            
        Returns:
            str: JSON string with extracted and enriched information
        """
        # Save the uploaded file
        file_path = create_or_get_upload_folder() / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Step 1: Extract data from the PDF
            print(f"Extracting data from PDF: {file.filename}")
            pdf_data_json = self.pdf_extractor.extract(file, query)
            
            # Parse the PDF data
            try:
                pdf_results = json.loads(pdf_data_json)
            except json.JSONDecodeError:
                print("Error parsing PDF extraction results as JSON")
                return pdf_data_json  # Return original result if cannot parse
            
            # Step 2: Enhance with web search
            print(f"Enhancing PDF data with web search")
            enhanced_results = self.web_search_agent.enhance_results(pdf_results)
            
            # Return the enhanced results as JSON
            return json.dumps(enhanced_results, indent=2)
            
        except Exception as e:
            import traceback
            print(f"Error in PDF web search extraction: {str(e)}")
            print(traceback.format_exc())
            return json.dumps({"error": str(e)})
        finally:
            file.file.close()