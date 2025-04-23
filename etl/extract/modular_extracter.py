from langchain_core.retrievers import BaseRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from etl.util.retrieval_util import create_retriever, PDFRetriever
from typing import Dict, List, Optional, Any, Union
import json
from etl.extract.abstract_extracter import AbstractExtracter
from fastapi import UploadFile, File

class ModularExtractor(AbstractExtracter):
    """
    A modular extractor that uses LangChain's retrieval patterns to extract
    information from various sources and formats it according to the requested schema.
    """
    
    def __init__(
        self, 
        model_name: str = "gpt-4o",
        enable_web_enrichment: bool = True,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the modular extractor.
        
        Args:
            model_name: The name of the LLM model to use
            enable_web_enrichment: Whether to enrich data from web sources
            chunk_size: Size of document chunks for splitting
            chunk_overlap: Overlap between chunks to maintain context
        """
        super().__init__()
        self.model_name = model_name
        self.enable_web_enrichment = enable_web_enrichment
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = ChatOpenAI(temperature=0, model=model_name)
    
    def extract(self, file: UploadFile, query: str = None) -> str:
        """
        Extract structured information from a file using LangChain's retrieval patterns.
        
        Args:
            file: The file to extract information from
            query: Custom extraction query (optional)
            
        Returns:
            JSON string with extracted information
        """
        # Save the uploaded file
        from etl.util.file_util import create_or_get_upload_folder
        import shutil
        import os
        
        file_path = create_or_get_upload_folder() / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        try:
            # Create document loader and retriever directly using LangChain's components
            from langchain_community.document_loaders import PyPDFLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_core.vectorstores import VectorStore
            from langchain_core.documents import Document
            from langchain_community.vectorstores import FAISS
            from langchain_openai import OpenAIEmbeddings
            
            # Load PDF documents
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len
            )
            
            chunked_documents = text_splitter.split_documents(documents)
            
            # Create vector store and retriever
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(chunked_documents, embeddings)
            pdf_retriever = vector_store.as_retriever()
            
            # Extract company name from PDF
            company_name = self._extract_company_name(pdf_retriever)
            
            # Create retrieval chain for PDF content
            pdf_qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=pdf_retriever,
                return_source_documents=True
            )
            
            # Extract information from PDF
            extraction_query = query or """
            Extract all company information, financial metrics, and operational 
            data from this document as a structured JSON object.
            """
            
            pdf_result = pdf_qa({"query": extraction_query})
            extracted_data = self._parse_qa_result(pdf_result["result"])
            
            # Enrich with web data if enabled and company name was found
            if self.enable_web_enrichment and company_name:
                # Create company info retriever
                company_retriever = create_retriever("company", company_name=company_name)
                
                # Create retrieval chain for company info
                company_qa = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=company_retriever,
                    return_source_documents=True
                )
                
                # Get company information
                company_query = f"Provide all financial and business information about {company_name}"
                company_result = company_qa({"query": company_query})
                
                # Merge data
                enriched_data = self._merge_data(
                    extracted_data, 
                    self._parse_qa_result(company_result["result"])
                )
                
                return json.dumps({
                    "main_category": enriched_data,
                    "company_name": company_name,
                    "source": "pdf+web"
                }, indent=2)
            else:
                # Return PDF data only
                return json.dumps({
                    "main_category": extracted_data,
                    "company_name": company_name,
                    "source": "pdf"
                }, indent=2)
                
        except Exception as e:
            import traceback
            print(f"Error in modular extraction: {str(e)}")
            print(traceback.format_exc())
            return json.dumps({"error": str(e)})
        finally:
            file.file.close()
    
    def _extract_company_name(self, pdf_retriever: BaseRetriever) -> Optional[str]:
        """
        Extract company name from PDF content.
        
        Args:
            pdf_retriever: The PDF retriever
            
        Returns:
            Extracted company name or None
        """
        try:
            from langchain.chains import LLMChain
            from langchain_core.prompts import PromptTemplate
            
            # Get first few chunks of the PDF
            docs = pdf_retriever.get_relevant_documents("company name")[:3]
            
            if not docs:
                return None
            
            # Combine text from the first few chunks
            text = "\n\n".join([doc.page_content for doc in docs])
            
            # Create a prompt to extract company name
            prompt = PromptTemplate(
                input_variables=["text"],
                template="""
                Extract the name of the company from the following text. 
                Provide ONLY the company name, nothing else.
                
                Text:
                {text}
                
                Company name:
                """
            )
            
            # Create and run chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            result = chain.run(text=text)
            
            # Clean up result
            return result.strip()
            
        except Exception as e:
            print(f"Error extracting company name: {str(e)}")
            return None
    
    def _parse_qa_result(self, result: str) -> Dict[str, Any]:
        """
        Parse the result from QA chain into a structured dictionary.
        
        Args:
            result: The result string from the QA chain
            
        Returns:
            Structured dictionary of extracted data
        """
        try:
            # Try to find and parse JSON in the result
            import json
            import re
            
            # Look for JSON in the result
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', result)
            
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                return json.loads(json_str)
            
            # If no JSON found, try to create structured data with the LLM
            from langchain.chains import LLMChain
            from langchain_core.prompts import PromptTemplate
            
            prompt = PromptTemplate(
                input_variables=["text"],
                template="""
                Convert the following text into a structured JSON object with appropriate keys and values:
                
                {text}
                
                Return ONLY valid JSON:
                """
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            structured_result = chain.run(text=result)
            
            # Try to parse the structured result
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', structured_result)
            
            if json_match:
                json_str = json_match.group(1) or json_match.group(2)
                return json.loads(json_str)
            
            return {"extracted_text": result}
            
        except Exception as e:
            print(f"Error parsing QA result: {str(e)}")
            return {"extracted_text": result}
    
    def _merge_data(self, pdf_data: Dict[str, Any], web_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge data from PDF and web sources, preferring PDF data when both are available.
        
        Args:
            pdf_data: Data extracted from PDF
            web_data: Data extracted from web sources
            
        Returns:
            Merged data dictionary
        """
        # Start with PDF data
        merged = pdf_data.copy()
        
        # Add web data for fields not in PDF data
        for key, value in web_data.items():
            if key not in merged or merged[key] is None:
                merged[key] = value
            elif isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_data(merged[key], value)
        
        return merged