from typing import Dict, Any, List, Optional, Union
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from etl.util.web_search_util import WebSearchUtils

class CompanyInfoRetriever(BaseRetriever):
    """
    A LangChain retriever that fetches company information from various sources.
    This allows for a unified interface to retrieve company data whether from PDFs or web sources.
    """
    
    def __init__(self, company_name: Optional[str] = None):
        """
        Initialize the company info retriever.
        
        Args:
            company_name: Optional name of the company to retrieve information for
        """
        super().__init__()  # Call parent init first
        self._company_name = company_name  # Use underscore prefix for private attribute
    
    @property
    def company_name(self) -> Optional[str]:
        """Getter for company_name"""
        return self._company_name
        
    @company_name.setter
    def company_name(self, value: str):
        """Setter for company_name"""
        self._company_name = value
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get documents related to the company based on the query.
        
        Args:
            query: The query string, which may contain the company name
            
        Returns:
            A list of Document objects with company information
        """
        # Extract company name from query if not provided in constructor
        company_name = self._company_name or self._extract_company_name(query)
        
        if not company_name:
            return []
        
        # Fetch data from various sources
        documents = []
        
        # Get basic company info
        try:
            company_info = WebSearchUtils.search_company_info(company_name)
            if company_info:
                documents.append(
                    Document(
                        page_content=f"Company Information for {company_name}:\n{str(company_info)}",
                        metadata={"source": "company_info", "company_name": company_name}
                    )
                )
        except Exception as e:
            print(f"Error fetching company info: {str(e)}")
        
        # Get financial data
        try:
            financial_data = WebSearchUtils.search_financial_data(company_name)
            if financial_data:
                documents.append(
                    Document(
                        page_content=f"Financial Information for {company_name}:\n{str(financial_data)}",
                        metadata={"source": "financial_data", "company_name": company_name}
                    )
                )
        except Exception as e:
            print(f"Error fetching financial data: {str(e)}")
        
        # Get news data
        try:
            news_data = WebSearchUtils.search_news(company_name)
            if news_data and "articles" in news_data:
                for article in news_data["articles"][:3]:  # Limit to top 3 articles
                    documents.append(
                        Document(
                            page_content=f"News Article: {article.get('title', '')}\n{article.get('description', '')}",
                            metadata={
                                "source": "news",
                                "company_name": company_name,
                                "article_url": article.get("url", ""),
                                "published_at": article.get("publishedAt", "")
                            }
                        )
                    )
        except Exception as e:
            print(f"Error fetching news data: {str(e)}")
        
        return documents
    
    def _extract_company_name(self, query: str) -> Optional[str]:
        """
        Extract company name from query string.
        
        Args:
            query: The query string
            
        Returns:
            Extracted company name or None
        """
        # Simple extraction - in a real implementation, this could use NER or more advanced techniques
        if "company:" in query.lower():
            parts = query.split("company:", 1)
            if len(parts) > 1:
                company_part = parts[1].strip()
                if " " in company_part:
                    return company_part.split(" ")[0]
                return company_part
        return None

class PDFRetriever(BaseRetriever):
    """
    A LangChain retriever that extracts information from PDF files.
    """
    
    def __init__(self, pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the PDF retriever.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks to split the PDF into
            chunk_overlap: Overlap between chunks to maintain context
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        super().__init__()
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Extract and retrieve relevant document chunks from the PDF.
        
        Args:
            query: The search query
            
        Returns:
            List of Document objects containing PDF text chunks
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            
            # Load the PDF using LangChain's document loader
            loader = PyPDFLoader(self.pdf_path)
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len
            )
            
            chunked_documents = text_splitter.split_documents(documents)
            
            # In a more advanced implementation, we could use embeddings to find the most relevant chunks
            # For now, just return all chunks
            return chunked_documents
            
        except Exception as e:
            print(f"Error loading PDF: {str(e)}")
            return []

# Factory function to create the appropriate retriever
def create_retriever(source_type: str, **kwargs) -> BaseRetriever:
    """
    Factory function to create the appropriate retriever based on source type.
    
    Args:
        source_type: Type of data source ("pdf", "company")
        **kwargs: Additional arguments for the specific retriever
        
    Returns:
        An instance of a BaseRetriever
    """
    if source_type == "pdf":
        if "pdf_path" not in kwargs:
            raise ValueError("pdf_path is required for PDF retriever")
        return PDFRetriever(pdf_path=kwargs["pdf_path"])
    
    elif source_type == "company":
        return CompanyInfoRetriever(company_name=kwargs.get("company_name"))
    
    else:
        raise ValueError(f"Unsupported source type: {source_type}")