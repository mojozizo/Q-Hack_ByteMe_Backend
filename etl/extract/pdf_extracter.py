import os
import shutil
import json
from pathlib import Path as PathLib

from fastapi import Path
from openai import OpenAI
from etl.extract.abstract_extracter import AbstractExtracter
from etl.util.file_util import create_or_get_upload_folder
from models.model import Category, PitchDeckMetricsAvailability
from models.metrics_evaluator import MetricsEvaluator

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PDFExtracter(AbstractExtracter):
    """Extracts structured data from PDF pitch decks using OpenAI."""

    def __init__(self):
        self.metrics_evaluator = MetricsEvaluator()
        self.metrics_availability = PitchDeckMetricsAvailability()

    def extract(self, file: Path, query: str) -> str:
        """
        Extracts structured information from a PDF pitch deck.
        
        Args:
            file: The uploaded PDF file
            query: Custom query for analysis (optional)
            
        Returns:
            str: Structured JSON response matching the Category model
        """
        # Save the uploaded file
        file_path = create_or_get_upload_folder() / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        try:
            # Process the PDF and get structured output
            return self._analyze_pdf(file_path, query)
        finally:
            file.file.close()

    def _analyze_pdf(self, pdf_path: PathLib, query: str = None) -> str:
        """Analyzes a PDF using OpenAI and returns structured data."""
        # Use default query if none provided
        prompt = "Analyze this pitch deck and extract key information." if not query else query
        
        # Get the schema for structured output
        schema = Category.model_json_schema()
        
        # Use assistants API to handle PDF upload and analysis
        with open(pdf_path, "rb") as file:
            # Upload the file
            uploaded_file = client.files.create(
                file=file,
                purpose="assistants"
            )
            
            # Create a simple assistant to analyze the PDF
            assistant = client.beta.assistants.create(
                name="PDF Analyzer",
                instructions=f"""Analyze the pitch deck and extract information according to this schema:
                {json.dumps(schema, indent=2)}
                
                Return ONLY valid JSON matching this schema exactly.
                """,
                model="gpt-4o",
                tools=[{"type": "file_search"}]
            )
            
            # Create a thread with the query
            thread = client.beta.threads.create()
            
            # Add a message with the file attachment
            client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt,
                attachments=[
                    {
                        "file_id": uploaded_file.id,
                        "tools": [{"type": "file_search"}]
                    }
                ]
            )
            
            # Run the analysis
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            
            # Wait for completion
            while True:
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                if run_status.status == "completed":
                    break
                elif run_status.status in ["failed", "cancelled", "expired"]:
                    raise Exception(f"Analysis failed: {run_status.status}")
                
            # Get the response
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            # Extract the response content
            response_text = ""
            for message in messages.data:
                if message.role == "assistant":
                    for content in message.content:
                        if content.type == "text":
                            response_text += content.text.value
            
            # Clean up resources
            client.files.delete(uploaded_file.id)
            client.beta.assistants.delete(assistant_id=assistant.id)
            
            # Try to parse and validate as JSON
            try:
                # Extract just the JSON part (in case there's additional text)
                import re
                json_match = re.search(r'```json\s*([\s\S]*?)\s*```|({[\s\S]*})', response_text)
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2)
                else:
                    json_str = response_text
                
                # Parse and validate with the Category model
                parsed = json.loads(json_str)
                validated_data = Category(**parsed)
                
                # Evaluate the metrics against benchmark values
                evaluation_results = self._evaluate_company_metrics(validated_data.model_dump())
                
                # Add metrics availability information
                metrics_availability = self._get_metrics_availability()
                
                # Combine the original data with evaluation results
                combined_result = {
                    "company_data": validated_data.model_dump(),
                    "evaluation": evaluation_results,
                    "metrics_availability": metrics_availability
                }
                
                return json.dumps(combined_result, indent=2)
            except Exception as e:
                # Return the raw response if parsing fails
                print(f"Failed to parse response as JSON: {str(e)}")
                return response_text

    def _evaluate_company_metrics(self, company_data: dict) -> dict:
        """
        Evaluate company metrics against benchmark values for different funding stages.
        
        Args:
            company_data: Dictionary containing extracted company data
            
        Returns:
            Dictionary with evaluation results
        """
        # Extract metrics that can be evaluated
        evaluable_metrics = {}
        
        # Map Category model fields to StartupMetrics fields
        field_mapping = {
            "hard_fund_criteria.annual_recurring_revenue": "annual_recurring_revenue",
            "financials.annual_recurring_revenues": "annual_recurring_revenue",
            "financials.revenues": "annual_recurring_revenue",  # Fallback
            "traction.customer_lifetime_value": "customer_lifetime_value",
            "financials.customer_acquisition_cost": "customer_acquisition_cost",
            "traction.conversion_rate": "conversion_rate",
            "financials.user_growth_rate": "user_growth_rate_yoy",
            "financials.revenue_growth_rate": "revenue_growth_rate_yoy",
            "financials.runway": "runway",
            "solution.time_to_value": "time_to_value",
            "financials.burn_multiple": "burn_multiple",
            "financials.monthly_cash_burn": "burn_rate",
            "business_model.margins": "gross_margin",
            "hard_fund_criteria.required_funding_amount": "required_funding_amount",
            "hard_fund_criteria.age_of_company": "age_of_company",
            "team.number_of_cofounders": "number_of_cofounders",
            "team.number_of_employees": "number_of_employees",
            "solution.ip_protected": "ip_protection",
            "traction.current_users": "monthly_active_users",
            "traction.stickiness": "product_stickiness",
            # New fields mapping
            "market.market_competitiveness": "market_competitiveness",
            "market.timing_score": "market_timing",
            "business_model.business_scalability": "business_model_scalability", 
            "financials.clean_cap_table": "cap_table_cleanliness",
            "roadmap.hiring_plan_aligned": "hiring_plan_alignment",
            "risks.regulatory_risks": "regulatory_risks",
            "risks.trend_risks": "trend_risks",
            "risks.litigation_risks": "litigation_risks",
            "risks.sanctions_check": "founder_sanction_status",
            "team.experience_in_branches": "founder_industry_experience",
            "team.past_exits": "founder_past_exits",
            "team.target_companies_universities": "founder_background"
        }
        
        # Extract metrics from nested company data
        for path, metric_name in field_mapping.items():
            parts = path.split('.')
            value = company_data
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
            
            if value is not None:
                evaluable_metrics[metric_name] = value
                
            # Check if this metric is unlikely to be found in a pitch deck
            metric_availability = getattr(self.metrics_availability, metric_name, 50)
            if metric_availability == 0 and metric_name not in evaluable_metrics:
                # For metrics that are almost never in pitch decks, we might want to use default values
                # or mark them as not applicable
                evaluable_metrics[metric_name] = None
        
        # Perform evaluation
        detailed_evaluation = self.metrics_evaluator.get_detailed_evaluation(evaluable_metrics)
        suitable_stages = self.metrics_evaluator.get_suitable_stages(evaluable_metrics)
        
        # Get metrics with low availability but high importance for funding decisions
        missing_key_metrics = self._get_missing_key_metrics(evaluable_metrics)
        
        return {
            "detailed_results": detailed_evaluation,
            "suitable_funding_stages": suitable_stages,
            "evaluated_metrics_count": len(evaluable_metrics),
            "total_metrics_count": len(self.metrics_evaluator.benchmarks.__dict__),
            "evaluation_coverage_percentage": round(len(evaluable_metrics) / len(self.metrics_evaluator.benchmarks.__dict__) * 100, 2),
            "missing_key_metrics": missing_key_metrics
        }
    
    def _get_missing_key_metrics(self, evaluable_metrics: dict) -> list:
        """
        Identify metrics that are important for funding decisions but often missing from pitch decks.
        
        Args:
            evaluable_metrics: Dictionary of metrics that were found in the pitch deck
            
        Returns:
            List of important metrics that were not found in the pitch deck
        """
        missing_metrics = []
        
        # Define metrics that are important for funding decisions
        key_metrics = [
            ("churn_rate", "Churn Rate"),
            ("net_revenue_retention", "Net Revenue Retention"),
            ("customer_payback_period", "Customer Payback Period"),
            ("burn_multiple", "Burn Multiple"),
            ("time_to_value", "Time to Value"),
            ("revenue_per_fte", "Revenue per FTE"),
            ("business_model_scalability", "Business Model Scalability")
        ]
        
        for metric_key, metric_name in key_metrics:
            if metric_key not in evaluable_metrics or evaluable_metrics[metric_key] is None:
                missing_metrics.append({
                    "metric_key": metric_key,
                    "metric_name": metric_name,
                    "importance": "high"
                })
                
        return missing_metrics
    
    def _get_metrics_availability(self) -> dict:
        """
        Get metrics availability information in a structured format.
        
        Returns:
            Dictionary with metrics availability information
        """
        availability_data = self.metrics_availability.model_dump()
        
        # Group metrics by availability
        grouped = {
            "not_available": [],
            "rarely_available": [],
            "sometimes_available": []
        }
        
        for metric, availability in availability_data.items():
            if availability == 0:
                grouped["not_available"].append(metric)
            elif availability == 1:
                grouped["rarely_available"].append(metric)
            else:
                grouped["sometimes_available"].append(metric)
                
        return grouped

    def get_category_schema(self):
        """Get the JSON schema for the Category model."""
        return Category.model_json_schema()