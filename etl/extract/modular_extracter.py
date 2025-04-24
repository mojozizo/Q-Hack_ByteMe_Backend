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
            
            # Store full text for later use in summarization
            full_text = "\n\n".join([doc.page_content for doc in documents])
            
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
            from langchain_core.runnables import RunnablePassthrough
            
            # Use invoke pattern instead of __call__
            retriever_chain = {"context": pdf_retriever, "query": RunnablePassthrough()}
            
            # Extract information from PDF
            extraction_query = query or """
            Extract all company information, financial metrics, and operational 
            data from this document as a structured JSON object.
            """
            
            # Format the data according to our flattened StartupMetrics model
            from models.model import StartupMetrics
            extraction_query += f"""
            Format the data exactly according to this schema to ensure compatibility:
            {StartupMetrics.model_json_schema()}
            
            Pay special attention to extracting founder information (names and roles) and include it 
            in the 'founders' field. Look for team sections, about us sections, or any mentions of 
            founders or leadership team.
            """
            
            # Use invoke instead of __call__
            from langchain_core.output_parsers import StrOutputParser
            pdf_qa = (
                {"query": RunnablePassthrough()} | 
                retriever_chain | 
                self._create_prompt_template() | 
                self.llm | 
                StrOutputParser()
            )
            
            pdf_result = pdf_qa.invoke(extraction_query)
            extracted_data = self._parse_qa_result(pdf_result)
            
            # Create a StartupMetrics instance
            try:
                metrics = StartupMetrics(**extracted_data)
                
                # Set company name if found
                if company_name and not metrics.company_name:
                    metrics.company_name = company_name
                
                # Extract a comprehensive pitch deck summary using the full text
                if not metrics.pitch_deck_summary:
                    metrics.pitch_deck_summary = self._generate_pitch_deck_summary(full_text)
                
                # Look for founder information if not already found
                if not metrics.founders:
                    metrics.founders = self._extract_founder_information(pdf_retriever)
                
                # Enrich with web data if enabled and company name was found
                if self.enable_web_enrichment and (company_name or metrics.company_name):
                    final_company_name = metrics.company_name or company_name
                    from etl.util.model_util import enrich_startup_metrics_from_web
                    enriched_metrics = enrich_startup_metrics_from_web(final_company_name, metrics)
                    
                    # For backward compatibility, keep the old format structure
                    business_info = {
                        "year_of_founding": enriched_metrics.year_of_founding,
                        "location_of_headquarters": enriched_metrics.location_of_headquarters,
                        "industry": enriched_metrics.industry,
                        "business_model": enriched_metrics.business_model,
                        "employees": enriched_metrics.employees,
                        "website_link": enriched_metrics.website_link,
                        "one_sentence_pitch": enriched_metrics.one_sentence_pitch
                    }
                    
                    financial_info = {
                        "annual_recurring_revenue": enriched_metrics.annual_recurring_revenue,
                        "monthly_recurring_revenue": enriched_metrics.monthly_recurring_revenue,
                        "customer_acquisition_cost": enriched_metrics.customer_acquisition_cost,
                        "customer_lifetime_value": enriched_metrics.customer_lifetime_value,
                        "cltv_cac_ratio": enriched_metrics.cltv_cac_ratio,
                        "gross_margin": enriched_metrics.gross_margin,
                        "revenue_growth_rates": None,
                        "monthly_active_users": enriched_metrics.monthly_active_users,
                        "sales_cycle_length": enriched_metrics.sales_cycle_length,
                        "burn_rate": enriched_metrics.burn_rate,
                        "runway": enriched_metrics.runway
                    }
                    
                    backward_compatibility = {
                        "main_category": {
                            "extracted_text": extracted_data.get("extracted_text", ""),
                            "business_information": business_info,
                            "financial_information": financial_info
                        },
                        "company_name": final_company_name,
                        "source": "pdf+web"
                    }
                    
                    # Return both the flattened and backward compatible data
                    return json.dumps({
                        **backward_compatibility,
                        "startup_metrics": enriched_metrics.model_dump()
                    }, indent=2)
                
                # Not using web enrichment, just return the extracted data
                business_info = {
                    "year_of_founding": metrics.year_of_founding,
                    "location_of_headquarters": metrics.location_of_headquarters,
                    "industry": metrics.industry,
                    "business_model": metrics.business_model,
                    "employees": metrics.employees,
                    "website_link": metrics.website_link,
                    "one_sentence_pitch": metrics.one_sentence_pitch
                }
                
                financial_info = {
                    "annual_recurring_revenue": metrics.annual_recurring_revenue,
                    "monthly_recurring_revenue": metrics.monthly_recurring_revenue,
                    "customer_acquisition_cost": metrics.customer_acquisition_cost,
                    "customer_lifetime_value": metrics.customer_lifetime_value,
                    "cltv_cac_ratio": metrics.cltv_cac_ratio,
                    "gross_margin": metrics.gross_margin,
                    "revenue_growth_rates": None,
                    "monthly_active_users": metrics.monthly_active_users,
                    "sales_cycle_length": metrics.sales_cycle_length,
                    "burn_rate": metrics.burn_rate,
                    "runway": metrics.runway
                }
                
                backward_compatibility = {
                    "main_category": {
                        "extracted_text": extracted_data.get("extracted_text", ""),
                        "business_information": business_info,
                        "financial_information": financial_info
                    },
                    "company_name": metrics.company_name or company_name,
                    "source": "pdf"
                }
                
                return json.dumps({
                    **backward_compatibility,
                    "startup_metrics": metrics.model_dump()
                }, indent=2)
                
            except Exception as e:
                print(f"Error processing metrics model: {str(e)}")
                # Fallback to original data
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
            
            # Get first few chunks of the PDF - using invoke() instead of get_relevant_documents()
            docs = pdf_retriever.invoke("company name")[:3]
            
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
            
            # Use prompt | llm pattern instead of LLMChain
            from langchain_core.runnables import RunnablePassthrough
            prompt_to_llm = {"text": RunnablePassthrough()} | prompt | self.llm
            result = prompt_to_llm.invoke(text)
            
            # Clean up result
            return result.content.strip()
            
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
                # Handle escape characters properly
                try:
                    # First attempt - direct parsing
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    # Second attempt - replace problematic escape sequences
                    json_str = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r'\\\\', json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Third attempt - use a more lenient approach
                        from langchain.chains import LLMChain
                        from langchain_core.prompts import PromptTemplate
                        
                        prompt = PromptTemplate(
                            input_variables=["text"],
                            template="""
                            The following text contains a JSON object with invalid escape sequences.
                            Fix the JSON and return a valid JSON object:
                            
                            {text}
                            
                            Return ONLY valid JSON without any markdown formatting:
                            """
                        )
                        
                        chain = LLMChain(llm=self.llm, prompt=prompt)
                        fixed_json = chain.run(text=json_str)
                        
                        # Clean and try to parse again
                        fixed_json = re.sub(r'```json\s*|\s*```', '', fixed_json)
                        return json.loads(fixed_json)
            
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

    def _create_prompt_template(self):
        """
        Create a prompt template for QA retrieval.
        
        Returns:
            A prompt template for use with LLM
        """
        from langchain_core.prompts import PromptTemplate
        
        template = """
        Use the following pieces of context to answer the question.
        If you don't know the answer based on the context, don't make it up - return a structured JSON
        with null values for fields you can't determine.
        
        Context:
        {context}
        
        Question:
        {query}
        
        Return a valid JSON object with your answer. Format it properly and use appropriate fields 
        based on the context.
        """
        
        return PromptTemplate.from_template(template)
    
    def _generate_pitch_deck_summary(self, full_text: str) -> str:
        """
        Generate a comprehensive summary of the entire pitch deck.
        
        Args:
            full_text: The complete text content of the PDF
            
        Returns:
            A detailed summary of the pitch deck
        """
        try:
            # Use a specialized prompt for summarization
            from langchain_core.prompts import PromptTemplate
            
            # Limit text length to avoid context window issues
            max_length = 15000
            summarization_text = full_text[:max_length] if len(full_text) > max_length else full_text
            
            prompt = PromptTemplate(
                input_variables=["text"],
                template="""
                You are a specialized pitch deck analyst. Create a comprehensive summary of this pitch deck.
                Focus on the key aspects including:
                
                1. Business concept and value proposition
                2. Market opportunity and target audience
                3. Product or service offering
                4. Business model and revenue streams
                5. Competitive landscape and advantages
                6. Traction and milestones achieved
                7. Team composition and expertise
                8. Financial highlights and projections
                9. Funding requirements and planned use of funds
                
                Provide a cohesive and detailed summary that captures the essence of the pitch deck.
                The summary should be 3-5 paragraphs long and highlight the most compelling aspects.
                
                Pitch Deck Content:
                {text}
                
                Summary:
                """
            )
            
            # Use prompt | llm pattern 
            from langchain_core.runnables import RunnablePassthrough
            prompt_to_llm = {"text": RunnablePassthrough()} | prompt | self.llm
            result = prompt_to_llm.invoke(summarization_text)
            
            return result.content.strip()
            
        except Exception as e:
            print(f"Error generating pitch deck summary: {str(e)}")
            return "No summary available."
    
    def _extract_founder_information(self, pdf_retriever: BaseRetriever) -> str:
        """
        Extract founder information (names and roles) from the pitch deck.
        
        Args:
            pdf_retriever: The PDF retriever
            
        Returns:
            String with founder names and roles
        """
        try:
            # Specifically search for sections likely to contain founder info
            founder_queries = [
                "team founders management leadership executives",
                "founder CEO CTO COO CFO chief executive officer",
                "co-founder cofounder founding team",
                "about us team our team leadership team executive team"
            ]
            
            # Get relevant sections for each query
            all_docs = []
            for query in founder_queries:
                docs = pdf_retriever.invoke(query)
                if docs:
                    all_docs.extend(docs)
            
            # Remove duplicates by page content
            unique_docs = {}
            for doc in all_docs:
                unique_docs[doc.page_content] = doc
            
            # If we have no relevant sections, return empty
            if not unique_docs:
                return None
            
            # Combine text from all unique sections
            text = "\n\n".join([doc.page_content for doc in unique_docs.values()])
            
            # Create a prompt to extract founder information
            from langchain_core.prompts import PromptTemplate
            
            prompt = PromptTemplate(
                input_variables=["text"],
                template="""
                Extract information about the founders or leadership team from the following text.
                Focus on identifying founders, co-founders, and key executives with their roles.
                Format the response as "Name (Role), Name (Role), ..." - for example:
                "John Smith (CEO & Co-founder), Jane Doe (CTO & Co-founder), Alex Johnson (CFO)"
                
                If no founder information is present, return null.
                
                Text:
                {text}
                
                Founder information:
                """
            )
            
            # Use prompt | llm pattern
            from langchain_core.runnables import RunnablePassthrough
            prompt_to_llm = {"text": RunnablePassthrough()} | prompt | self.llm
            result = prompt_to_llm.invoke(text)
            
            founder_info = result.content.strip()
            
            # If the response is explicitly "null" or similar, return None
            if founder_info.lower() in ["null", "none", "not found", "no founder information"]:
                return None
                
            return founder_info
            
        except Exception as e:
            print(f"Error extracting founder information: {str(e)}")
            return None