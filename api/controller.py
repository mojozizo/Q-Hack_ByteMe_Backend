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
            
            # If we have a startup_metrics field in the response, use that directly
            import json
            if isinstance(extracted_data, str):
                try:
                    data_dict = json.loads(extracted_data)
                    if "startup_metrics" in data_dict:
                        # Use the already formatted startup_metrics data
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
                except:
                    pass
            
            # Format the response to match expected output structure
            if isinstance(extracted_data, str):
                try:
                    extracted_data = json.loads(extracted_data)
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
            
            # Now create a StartupMetrics instance from the data
            from models.model import StartupMetrics
            
            # Initialize with defaults
            metrics = StartupMetrics()
            
            # Check if we have the nested format with business_information
            if isinstance(extracted_data, dict) and "main_category" in extracted_data:
                main_cat = extracted_data["main_category"]
                
                # Extract company name
                if "company_name" in extracted_data:
                    metrics.company_name = extracted_data["company_name"]
                
                # Extract from business_information
                if "business_information" in main_cat:
                    bus_info = main_cat["business_information"]
                    
                    if "year_of_founding" in bus_info:
                        metrics.year_of_founding = bus_info["year_of_founding"]
                    
                    if "location_of_headquarters" in bus_info:
                        metrics.location_of_headquarters = bus_info["location_of_headquarters"]
                    
                    if "industry" in bus_info:
                        metrics.industry = bus_info["industry"]
                    
                    if "business_model" in bus_info:
                        metrics.business_model = bus_info["business_model"]
                    
                    if "employees" in bus_info:
                        metrics.employees = bus_info["employees"]
                    
                    if "website_link" in bus_info:
                        metrics.website_link = bus_info["website_link"]
                    
                    if "one_sentence_pitch" in bus_info:
                        metrics.one_sentence_pitch = bus_info["one_sentence_pitch"]
                
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
                
                # Return combined data
                processed_info = json.dumps({
                    "main_category": main_cat,
                    "company_name": metrics.company_name,
                    "source": extracted_data.get("source", "pdf"),
                    "flattened_metrics": metrics.model_dump()
                })
            else:
                # Not in the expected format, just return as is
                processed_info = json.dumps(extracted_data)
            
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