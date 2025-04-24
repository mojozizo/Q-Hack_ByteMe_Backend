from typing import Dict, Type, Any, Optional, Union, get_type_hints
from pydantic import BaseModel, Field, create_model

def discover_nested_models(model_class: Type[BaseModel]) -> Dict[str, Type[BaseModel]]:
    """
    Discovers nested Pydantic models within a model class.
    
    Args:
        model_class: The Pydantic model class to inspect
        
    Returns:
        Dict mapping field names to their nested model classes
    """
    nested_fields = {}
    # Check type annotations for nested models
    type_hints = get_type_hints(model_class)
    
    for field_name, field_type in type_hints.items():
        # Handle Optional[SomeModel]
        if hasattr(field_type, "__origin__") and field_type.__origin__ is Union:
            # Check if this is Optional[SomeModel]
            args = field_type.__args__
            if len(args) == 2 and type(None) in args:
                # It's an Optional - check if the other type is a BaseModel
                other_type = args[0] if args[1] is type(None) else args[1]
                if hasattr(other_type, "model_fields") and issubclass(other_type, BaseModel):
                    nested_fields[field_name] = other_type
        
        # Handle direct BaseModel fields
        elif hasattr(field_type, "model_fields") and issubclass(field_type, BaseModel):
            nested_fields[field_name] = field_type
    
    return nested_fields

def generate_extraction_prompt(model_class: Type[BaseModel]) -> str:
    """
    Generates an extraction prompt based on a Pydantic model's structure.
    
    Args:
        model_class: The Pydantic model class to use as template
        
    Returns:
        A string prompt suitable for extracting data matching the model
    """
    prompt = "Extract the following information from this document:\n\n"
    
    # Get nested models first
    nested_fields = discover_nested_models(model_class)
    
    # Add fields from the main model
    field_idx = 1
    for field_name, field in model_class.model_fields.items():
        # Skip nested model fields as they'll be handled separately
        if field_name in nested_fields:
            continue
            
        field_desc = field.description or field_name.replace('_', ' ').title()
        field_type = "string"
        
        # Try to determine field type
        if field.annotation:
            if "int" in str(field.annotation).lower():
                field_type = "integer"
            elif "float" in str(field.annotation).lower() or "decimal" in str(field.annotation).lower():
                field_type = "number"
            elif "bool" in str(field.annotation).lower():
                field_type = "boolean (use true/false)"
            elif "date" in str(field.annotation).lower():
                field_type = "date (YYYY-MM-DD format)"
        
        prompt += f"{field_idx}. {field_desc} as a {field_type}\n"
        field_idx += 1
    
    # Add nested model fields
    for nested_field_name, nested_model in nested_fields.items():
        prompt += f"\nFor {nested_field_name.replace('_', ' ').title()}, extract:\n"
        
        for j, (sub_field_name, sub_field) in enumerate(nested_model.model_fields.items(), 1):
            sub_desc = sub_field.description or sub_field_name.replace('_', ' ').title()
            sub_type = "string"
            
            # Try to determine field type
            if sub_field.annotation:
                if "int" in str(sub_field.annotation).lower():
                    sub_type = "integer"
                elif "float" in str(sub_field.annotation).lower() or "decimal" in str(sub_field.annotation).lower():
                    sub_type = "number"
                elif "bool" in str(sub_field.annotation).lower():
                    sub_type = "boolean (use true/false)"
                elif "date" in str(sub_field.annotation).lower():
                    sub_type = "date (YYYY-MM-DD format)"
            
            prompt += f"  {j}. {sub_desc} as a {sub_type}\n"
    
    prompt += "\nProvide data ONLY for these specific fields and in the exact format requested. "
    prompt += "Return the data as a valid JSON object that follows the schema structure."
    
    return prompt

def generate_assistant_instructions(model_class: Type[BaseModel]) -> str:
    """
    Generates instructions for OpenAI Assistant based on a Pydantic model.
    
    Args:
        model_class: The Pydantic model class to use as template
        
    Returns:
        A string with instructions for the OpenAI Assistant
    """
    # Get the main schema
    schema = model_class.model_json_schema()
    
    # Get nested schemas
    nested_fields = discover_nested_models(model_class)
    nested_schemas = {field_name: model.model_json_schema() for field_name, model in nested_fields.items()}
    
    # Build the instructions
    instructions = f"""You are a specialized document analyzer. 
    Analyze the document and extract ONLY the information specified in this schema:
    
    Main Schema:
    {schema}
    """
    
    # Add nested schema instructions if any
    for field_name, nested_schema in nested_schemas.items():
        instructions += f"""
        
        {field_name} Schema (to be nested within the main schema):
        {nested_schema}
        """
    
    instructions += """
    
    Important guidelines:
    1. All numeric values should be integers only (e.g., 15.5% becomes 16)
    2. For boolean values, use 1 for yes/true and 0 for no/false when the field is typed as integer
    3. For scale metrics, use values from 1-5 where specified
    4. Only include fields defined in the schema - do not add extra fields
    5. Return ONLY valid JSON matching the schema exactly
    6. Do not include any narrative analysis or additional text
    7. For any field you cannot find information for, leave it out (don't include null/empty values)
    """
    
    return instructions

def create_dynamic_model(model_name: str, fields_dict: Dict[str, Any]) -> Type[BaseModel]:
    """
    Creates a Pydantic model dynamically from a dictionary of field definitions.
    
    Args:
        model_name: Name for the new model
        fields_dict: Dictionary mapping field names to their type and description
        
    Returns:
        A new Pydantic model class
    
    Example:
        fields = {
            "name": (str, Field(description="Person's name")),
            "age": (int, Field(description="Person's age"))
        }
        PersonModel = create_dynamic_model("Person", fields)
    """
    return create_model(model_name, **fields_dict)

def enrich_model_from_web(
    model_instance: BaseModel,
    company_name_field: str = "company_name",
    web_search_util = None
) -> BaseModel:
    """
    Enriches a model instance with web data if available.
    
    Args:
        model_instance: The Pydantic model instance to enrich
        company_name_field: The field name that contains the company name
        web_search_util: The WebSearchUtils class to use (optional)
        
    Returns:
        The enriched model instance
    """
    # Import WebSearchUtils if not provided
    if web_search_util is None:
        from etl.util.web_search_util import WebSearchUtils
        web_search_util = WebSearchUtils
    
    # Get company name from the model if available
    company_name = None
    
    # Try to find company name in any field that might contain it 
    for field_name in model_instance.model_fields.keys():
        if company_name_field.lower() in field_name.lower() and hasattr(model_instance, field_name):
            field_value = getattr(model_instance, field_name, None)
            if field_value and isinstance(field_value, str):
                company_name = field_value
                break
    
    # If we found a company name, try to enrich the data
    if company_name:
        # Get company info from web
        company_data = web_search_util.search_company_info(company_name)
        
        # Update model fields with company data
        for api_field, value in company_data.items():
            if hasattr(model_instance, api_field) and not getattr(model_instance, api_field, None):
                setattr(model_instance, api_field, value)
        
        # Try to get financial data
        financial_data = web_search_util.search_financial_data(company_name)
        for field_name, value in financial_data.items():
            if hasattr(model_instance, field_name) and not getattr(model_instance, field_name, None):
                setattr(model_instance, field_name, value)
                
        # Get additional metrics data
        category_data = web_search_util.search_category_to_search_data(company_name)
        for field_name, value in category_data.items():
            if hasattr(model_instance, field_name) and not getattr(model_instance, field_name, None):
                setattr(model_instance, field_name, value)
    
    return model_instance

def enrich_startup_metrics_from_web(
    company_name: str,
    existing_metrics_model: Optional[BaseModel] = None,
    web_search_util = None
) -> BaseModel:
    """
    Creates and populates a StartupMetrics model instance with all available metrics.
    
    Args:
        company_name: The name of the company to search for
        existing_metrics_model: Optional existing StartupMetrics model with some data
        web_search_util: The WebSearchUtils class to use (optional)
        
    Returns:
        A populated StartupMetrics model instance with all available data
    """
    # Import required modules
    from models.model import StartupMetrics
    
    if web_search_util is None:
        from etl.util.web_search_util import WebSearchUtils
        web_search_util = WebSearchUtils
    
    # Create a new model or use existing one
    metrics = existing_metrics_model or StartupMetrics(company_name=company_name)
    
    try:
        # Get company information
        company_data = web_search_util.search_company_info(company_name)
        
        # Get financial metrics
        financial_data = web_search_util.search_financial_data(company_name)
        
        # Get advanced metrics data
        advanced_data = web_search_util.search_category_to_search_data(company_name)
        
        # Update all fields in the model
        for data_source in [company_data, financial_data, advanced_data]:
            for field_name, value in data_source.items():
                if hasattr(metrics, field_name) and not getattr(metrics, field_name, None):
                    setattr(metrics, field_name, value)
        
        return metrics
        
    except Exception as e:
        print(f"Error enriching StartupMetrics: {str(e)}")
        return metrics

# For backward compatibility
enrich_category_to_search = enrich_startup_metrics_from_web