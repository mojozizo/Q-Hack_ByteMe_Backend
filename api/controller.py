from fastapi import UploadFile, File, APIRouter, HTTPException, Path, Query
from starlette.responses import JSONResponse
import json

from etl.extract.extractor_handler import ExtractorHandler
from etl.util.file_util import create_or_get_upload_folder
from etl.agent.news_agent import NewsAgent
from etl.agent.linkedin_agent import LinkedInAgent
from etl.agent.orchestrator_agent import OrchestratorAgent
from etl.agent.financial_agent import FinancialAgent

router = APIRouter()

@router.post("/upload-pdf/")
async def upload_pdf(
    file: UploadFile = File(...), 
    query: str = None,
    use_agent_workflow: bool = Query(False, description="Whether to use LangChain agent workflow"),
    use_modular_workflow: bool = Query(False, description="Whether to use the new modular retrieval workflow")
):
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"message": "Only PDF files are allowed"}
            )

        file_path = create_or_get_upload_folder() / file.filename
        
        try:
            # Process the file and get the extracted data
            extracted_data = ExtractorHandler.get_extractor(
                "pdf", 
                use_agent_workflow=use_agent_workflow,
                use_modular_workflow=use_modular_workflow
            ).extract(file, query)
            
            # Process the response to match the required format
            import json
            
            # If extracted_data is a string, try to parse it as JSON
            if isinstance(extracted_data, str):
                try:
                    extracted_data_dict = json.loads(extracted_data)
                except:
                    # If it can't be parsed as JSON, just return it as is
                    return JSONResponse(
                        status_code=200,
                        content={
                            "message": "File uploaded and processed successfully",
                            "filename": file.filename,
                            "file_path": str(file_path),
                            "processed_info": extracted_data,
                            "used_agent_workflow": use_agent_workflow,
                            "used_modular_workflow": use_modular_workflow
                        }
                    )
            else:
                extracted_data_dict = extracted_data
            
            # Check if we have the startup_metrics field in the response
            if isinstance(extracted_data_dict, dict) and "startup_metrics" in extracted_data_dict:
                metrics_dict = extracted_data_dict["startup_metrics"]
                
                # Get all metrics data if it's already structured
                if isinstance(metrics_dict, dict):
                    # Define company_info fields (summary fields)
                    company_info_fields = [
                        "company_name", "official_company_name", "year_of_founding",
                        "location_of_headquarters", "business_model", "industry",
                        "required_funding_amount", "employees", "website_link",
                        "one_sentence_pitch", "linkedin_profile_ceo", "pitch_deck_summary"
                    ]
                    
                    # Define the criteria fields to be placed at the top level
                    criteria_fields = [
                        "annual_recurring_revenue", "monthly_recurring_revenue", 
                        "customer_acquisition_cost", "customer_lifetime_value",
                        "cltv_cac_ratio", "gross_margin", "revenue_growth_rate_yoy",
                        "revenue_growth_rate_mom", "sales_cycle_length", "monthly_active_users",
                        "user_growth_rate_yoy", "user_growth_rate_mom", "conversion_rate",
                        "pricing_strategy_maturity", "burn_rate", "runway", "ip_protection", 
                        "market_competitiveness", "market_timing", "cap_table_cleanliness",
                        "founder_industry_experience", "founder_past_exits", "founder_background",
                        "country_of_headquarters"
                    ]
                    
                    # Extract company_info from the existing structure
                    company_info = {}
                    for field in company_info_fields:
                        # First check if the field is in company_info, then check the top level
                        if "company_info" in metrics_dict and field in metrics_dict["company_info"]:
                            company_info[field] = metrics_dict["company_info"][field]
                        elif field in metrics_dict:
                            company_info[field] = metrics_dict[field]
                        else:
                            company_info[field] = None
                    
                    # Create the final response structure
                    response_data = {
                        "company_info": company_info
                    }
                    
                    # Add all criteria fields directly at the top level
                    for field in criteria_fields:
                        # If we have a nested structure, extract from it
                        if field in metrics_dict:
                            response_data[field] = metrics_dict[field]
                        elif "financial_info" in metrics_dict and field in metrics_dict["financial_info"]:
                            response_data[field] = metrics_dict["financial_info"][field]
                        else:
                            response_data[field] = None
                    
                    # Update the extracted_data_dict with our new format
                    extracted_data_dict["startup_metrics"] = response_data
                
                # Convert back to JSON string
                processed_info = json.dumps(extracted_data_dict)
            else:
                # Format the response based on our general implementation
                # Create a StartupMetrics instance from the data
                from models.model import StartupMetrics
                
                # Initialize with defaults
                metrics = StartupMetrics()
                
                # Check if we have the nested format with business_information
                if isinstance(extracted_data_dict, dict) and "main_category" in extracted_data_dict:
                    main_cat = extracted_data_dict["main_category"]
                    
                    # Extract company name
                    if "company_name" in extracted_data_dict:
                        metrics.company_name = extracted_data_dict["company_name"]
                    
                    # Extract from business_information
                    if "business_information" in main_cat:
                        bus_info = main_cat["business_information"]
                        
                        for field in ["year_of_founding", "location_of_headquarters", "industry", 
                                    "business_model", "employees", "website_link", "one_sentence_pitch"]:
                            if field in bus_info:
                                setattr(metrics, field, bus_info[field])
                    
                    # Extract from financial_information
                    if "financial_information" in main_cat:
                        fin_info = main_cat["financial_information"]
                        
                        # Map financial fields
                        financial_fields = [
                            "annual_recurring_revenue", "monthly_recurring_revenue",
                            "customer_acquisition_cost", "customer_lifetime_value",
                            "cltv_cac_ratio", "gross_margin", "monthly_active_users",
                            "sales_cycle_length", "burn_rate", "runway"
                        ]
                        
                        for field in financial_fields:
                            if field in fin_info and fin_info[field] is not None:
                                setattr(metrics, field, fin_info[field])
                    
                    # Get pitch deck summary from extracted_text
                    if "extracted_text" in main_cat and main_cat["extracted_text"]:
                        metrics.pitch_deck_summary = main_cat["extracted_text"]
                    
                    # Create response in the expected structure
                    metrics_dict = metrics.model_dump()
                    
                    # Define company_info fields (summary fields)
                    company_info_fields = [
                        "company_name", "official_company_name", "year_of_founding",
                        "location_of_headquarters", "business_model", "industry",
                        "required_funding_amount", "employees", "website_link",
                        "one_sentence_pitch", "linkedin_profile_ceo", "pitch_deck_summary"
                    ]
                    
                    # Create the company_info object
                    company_info = {field: metrics_dict.get(field) for field in company_info_fields}
                    
                    # Define the criteria fields to be placed at top level
                    criteria_fields = [
                        "annual_recurring_revenue", "monthly_recurring_revenue", 
                        "customer_acquisition_cost", "customer_lifetime_value",
                        "cltv_cac_ratio", "gross_margin", "revenue_growth_rate_yoy",
                        "revenue_growth_rate_mom", "sales_cycle_length", "monthly_active_users",
                        "user_growth_rate_yoy", "user_growth_rate_mom", "conversion_rate",
                        "pricing_strategy_maturity", "burn_rate", "runway", "ip_protection", 
                        "market_competitiveness", "market_timing", "cap_table_cleanliness",
                        "founder_industry_experience", "founder_past_exits", "founder_background",
                        "country_of_headquarters"
                    ]
                    
                    # Create the final response structure
                    response_data = {
                        "company_info": company_info
                    }
                    
                    # Add all criteria fields directly at the top level
                    for field in criteria_fields:
                        response_data[field] = metrics_dict.get(field)
                    
                    # Update extracted_data_dict with our formatted metrics
                    extracted_data_dict["startup_metrics"] = response_data
                    
                    # Convert back to JSON string
                    processed_info = json.dumps(extracted_data_dict)
                else:
                    # Not in the expected format, just return as is
                    processed_info = json.dumps(extracted_data_dict)
            
            return JSONResponse(
                status_code=200,
                content={
                    "message": "File uploaded and processed successfully",
                    "filename": file.filename,
                    "file_path": str(file_path),
                    "processed_info": processed_info,
                    "used_agent_workflow": use_agent_workflow,
                    "used_modular_workflow": use_modular_workflow
                }
            )
        except Exception as processing_error:
            # Log detailed error for debugging
            import traceback
            print(f"PDF processing error: {str(processing_error)}")
            print(traceback.format_exc())
            
            return JSONResponse(
                status_code=500,
                content={
                    "message": f"Error processing PDF content: {str(processing_error)}",
                    "filename": file.filename,
                    "file_path": str(file_path)
                }
            )
    except Exception as e:
        # Handle general exceptions
        import traceback
        print(f"General error: {str(e)}")
        print(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={"message": f"Server error: {str(e)}"}
        )
    finally:
        # Always close the file
        file.file.close()


@router.get("/news/{company_name}")
async def get_company_news(company_name: str):
    """
    Get news data for a specific company.
    
    Args:
        company_name: The name of the company to get news for
        
    Returns:
        News data about the company
    """
    try:
        # Initialize the news agent
        news_agent = NewsAgent()
        
        # Get news about the company
        news_data = news_agent._run(company_name)
        
        # Convert the Pydantic model to a dictionary for JSON serialization
        if hasattr(news_data, "model_dump"):
            news_data_dict = news_data.model_dump()
        elif hasattr(news_data, "dict"):
            # For backwards compatibility with older Pydantic versions
            news_data_dict = news_data.dict()
        else:
            # If it's not a Pydantic model, use it directly
            news_data_dict = news_data
        
        # Return the news data
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully retrieved news data for {company_name}",
                "company_name": company_name,
                "news_data": news_data_dict
            }
        )
    except Exception as e:
        # Handle exceptions
        import traceback
        print(f"Error retrieving news data: {str(e)}")
        print(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={"message": f"Error retrieving news data: {str(e)}"}
        )


@router.get("/linkedin/{profile_url:path}")
async def get_linkedin_profile(profile_url: str):
    """
    Get LinkedIn profile data for a specific URL.
    
    Args:
        profile_url: The LinkedIn profile URL to analyze
        
    Returns:
        Structured data about the LinkedIn profile
    """
    try:
        # Check if the URL is a valid LinkedIn URL
        if not profile_url.startswith("https://www.linkedin.com/") and not profile_url.startswith("linkedin.com/"):
            return JSONResponse(
                status_code=400,
                content={"message": "Invalid LinkedIn URL format. Please provide a valid LinkedIn profile URL."}
            )
        
        # Initialize the LinkedIn agent
        linkedin_agent = LinkedInAgent()
        
        # Get LinkedIn profile data
        linkedin_data = linkedin_agent._run(profile_url)
        
        # If the data is a JSON string, parse it
        if isinstance(linkedin_data, str):
            try:
                linkedin_data_dict = json.loads(linkedin_data)
            except json.JSONDecodeError:
                linkedin_data_dict = {"raw_data": linkedin_data}
        else:
            linkedin_data_dict = linkedin_data
        
        # Return the LinkedIn data
        return JSONResponse(
            status_code=200,
            content={
                "message": "Successfully retrieved LinkedIn profile data",
                "profile_url": profile_url,
                "linkedin_data": linkedin_data_dict
            }
        )
    except Exception as e:
        # Handle exceptions
        import traceback
        print(f"Error retrieving LinkedIn data: {str(e)}")
        print(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={"message": f"Error retrieving LinkedIn data: {str(e)}"}
        )


@router.post("/orchestrate/")
async def orchestrate_analysis(
    file: UploadFile = File(...),
    query: str = None
):
    """
    Orchestrate a complete analysis using all agents.
    
    Args:
        file: The uploaded PDF file to process
        query: Optional query to guide the extraction
        
    Returns:
        Consolidated results from all agents
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"message": "Only PDF files are allowed"}
            )
        
        # Initialize the orchestrator
        orchestrator = OrchestratorAgent()
        
        # Process the file through the orchestrator
        orchestrator_output = orchestrator.extract(file, query)
        
        # Convert JSON string to dictionary
        try:
            consolidated_results = json.loads(orchestrator_output)
        except json.JSONDecodeError:
            print(f"Error parsing orchestrator output: {orchestrator_output}")
            return JSONResponse(
                status_code=500,
                content={"message": "Error parsing orchestrator output", "raw_output": orchestrator_output}
            )
        
        # Create a properly formatted response from the model
        from models.model import StartupMetrics
        
        # Initialize with default values
        formatted_metrics = StartupMetrics()
        
        # Check if we have a main_category structure in the results
        if "main_category" in consolidated_results:
            main_category = consolidated_results["main_category"]
            
            # First, extract company_info separately if it exists
            company_info = {}
            if "company_info" in main_category and main_category["company_info"]:
                company_info = main_category["company_info"]
                
                # Extract company info fields
                company_info_fields = [
                    "company_name", "official_company_name", "year_of_founding",
                    "location_of_headquarters", "business_model", "industry",
                    "required_funding_amount", "employees", "website_link",
                    "one_sentence_pitch", "linkedin_profile_ceo", "pitch_deck_summary"
                ]
                
                for field in company_info_fields:
                    if field in company_info and company_info[field] is not None:
                        setattr(formatted_metrics, field, company_info[field])
            
            # Then check for fields directly in main_category
            # For each field in the StartupMetrics model
            for field in formatted_metrics.model_fields:
                # First check if it exists directly in main_category
                if field in main_category and main_category[field] is not None:
                    setattr(formatted_metrics, field, main_category[field])
                # If not in main_category, check if it's in company_info but wasn't processed yet
                elif field in company_info and company_info[field] is not None:
                    setattr(formatted_metrics, field, company_info[field])
            
            # If company_name wasn't found, check a few more places
            if not formatted_metrics.company_name:
                if "company_name" in main_category and main_category["company_name"]:
                    formatted_metrics.company_name = main_category["company_name"]
                elif "search_category" in consolidated_results and "company_name" in consolidated_results["search_category"]:
                    formatted_metrics.company_name = consolidated_results["search_category"]["company_name"]
                
        # Add data from other sources (LinkedIn, News) if they exist in consolidated_results
        if "linkedin_data" in consolidated_results and consolidated_results["linkedin_data"]:
            linkedin_data = consolidated_results["linkedin_data"]
            
            # Map LinkedIn data to appropriate fields
            if not formatted_metrics.founder_linkedin_summary and "summary" in linkedin_data:
                formatted_metrics.founder_linkedin_summary = linkedin_data["summary"]
                
            if not formatted_metrics.founder_skills and "skills" in linkedin_data:
                formatted_metrics.founder_skills = linkedin_data["skills"]
                
            if not formatted_metrics.founder_linkedin_url and "source_url" in linkedin_data:
                formatted_metrics.founder_linkedin_url = linkedin_data["source_url"]
                
            # Add more LinkedIn mappings as needed
        
        # Map news data to appropriate fields
        if "news_data" in consolidated_results and consolidated_results["news_data"]:
            news_data = consolidated_results["news_data"]
            
            # Map news sentiment if available
            if not formatted_metrics.news_sentiment and "tone" in news_data:
                tone = news_data["tone"].lower() if isinstance(news_data["tone"], str) else ""
                if "positive" in tone:
                    formatted_metrics.news_sentiment = "positive"
                elif "negative" in tone:
                    formatted_metrics.news_sentiment = "negative"
                else:
                    formatted_metrics.news_sentiment = "neutral"
                    
            if not formatted_metrics.recent_news_summary and "summary" in news_data:
                formatted_metrics.recent_news_summary = news_data["summary"]
        
        # Include original data for debugging
        formatted_response = {
            "message": "File processed successfully with orchestrator",
            "filename": file.filename,
            "startup_metrics": formatted_metrics.model_dump()
        }
        
        # Include original consolidated results for compatibility
        formatted_response["raw_results"] = consolidated_results
        
        return JSONResponse(
            status_code=200,
            content=formatted_response
        )
    except Exception as e:
        # Handle exceptions
        import traceback
        print(f"Orchestrator error: {str(e)}")
        print(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={"message": f"Orchestrator error: {str(e)}"}
        )
    finally:
        # Always close the file
        file.file.close()


@router.get("/financial/{company_name}")
async def get_company_financial(company_name: str):
    """
    Get financial data for a specific company from SEC EDGAR.
    
    Args:
        company_name: The name of the company or CIK number to get financial data for
        
    Returns:
        Financial data about the company from SEC EDGAR
    """
    try:
        # Initialize the financial agent
        financial_agent = FinancialAgent()
        
        # Get financial data about the company
        financial_data = financial_agent._run(company_name)
        
        # If the data is a JSON string, parse it
        if isinstance(financial_data, str):
            try:
                financial_data_dict = json.loads(financial_data)
            except json.JSONDecodeError:
                financial_data_dict = {"raw_data": financial_data}
        else:
            financial_data_dict = financial_data
        
        # Return the financial data
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully retrieved financial data for {company_name}",
                "company_name": company_name,
                "financial_data": financial_data_dict
            }
        )
    except Exception as e:
        # Handle exceptions
        import traceback
        print(f"Error retrieving financial data: {str(e)}")
        print(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={"message": f"Error retrieving financial data: {str(e)}"}
        )


@router.get("/")
async def root():
    return {"message": "Welcome to ByteMe"}