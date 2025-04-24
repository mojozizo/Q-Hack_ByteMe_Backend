from fastapi import UploadFile, File, APIRouter, HTTPException, Path, Query
from starlette.responses import JSONResponse

from etl.extract.extractor_handler import ExtractorHandler
from etl.util.file_util import create_or_get_upload_folder

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


@router.get("/")
async def root():
    return {"message": "Welcome to ByteMe"}