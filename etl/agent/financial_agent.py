import json
import os
import re
import traceback

from langchain_openai import ChatOpenAI

from etl.transform.parsers.sec_edgar_parser import SecEdgarParser
from etl.util.web_search_util import WebSearchUtils
from models.financial_model import FinancialModel, FilingModel


class FinancialAgent:
    """Agent for SEC EDGAR financial data extraction."""

    def __init__(self):
        """Initialize the Financial agent with OpenAI client."""
        self.name = "Financial Agent"
        # Initialize the LLM
        self.llm = ChatOpenAI(
            temperature=0.3,
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY")
        )

    def _run(self, input_company: str) -> str:
        """
        Run the agent with the given company name or CIK.

        Args:
            input_company: Company name or CIK number

        Returns:
            Structured JSON data of the financial information
        """
        if input_company.strip().isdigit() or (input_company.strip().startswith('0') and input_company.strip()[1:].isdigit()):
            cik = input_company.strip().zfill(10)
        else:
            cik = WebSearchUtils.search_cik_by_name(input_company)
            if not cik:
                return json.dumps({
                    "error": f"Could not find CIK for company: {input_company}"
                })

        # Use the SecEdgarParser to parse the CIK
        parser = SecEdgarParser(cik=cik)
        parsed_data = parser.parse()

        try:
            all_facts = parser.fetch_all_facts()
            parsed_data["facts"] = all_facts
        except Exception as e:
            parsed_data["facts_error"] = str(e)

        processed_data = self.process_parsed_data(parsed_data)

        return processed_data

    def process_parsed_data(self, parsed_data: dict) -> str:
        """Process parsed financial data with LLM using severely trimmed data."""
        try:
            minimal_data = {}

            if "name" in parsed_data:
                minimal_data["name"] = parsed_data["name"]
            if "cik" in parsed_data:
                minimal_data["cik"] = parsed_data["cik"]
            if "sic" in parsed_data:
                minimal_data["sic"] = parsed_data["sic"]
            if "sicDescription" in parsed_data:
                minimal_data["sicDescription"] = parsed_data["sicDescription"]

            if "filings" in parsed_data and "recent" in parsed_data["filings"]:
                minimal_data["filings"] = {"recent": parsed_data["filings"]["recent"][:2]}

            if "facts" in parsed_data and isinstance(parsed_data["facts"], dict):
                minimal_facts = {}
                if "us-gaap" in parsed_data["facts"] and isinstance(parsed_data["facts"]["us-gaap"], dict):
                    us_gaap = {}
                    key_metrics = ["Revenue", "Revenues", "NetIncomeLoss", "Assets", "Liabilities"]

                    for metric in key_metrics:
                        if metric in parsed_data["facts"]["us-gaap"]:
                            metric_data = parsed_data["facts"]["us-gaap"][metric]
                            if isinstance(metric_data, dict) and "units" in metric_data:
                                units = metric_data["units"]
                                trimmed_units = {}
                                for unit_type, values in units.items():
                                    if isinstance(values, list) and values:
                                        trimmed_units[unit_type] = [values[-1]]
                                us_gaap[metric] = {"units": trimmed_units}

                    if us_gaap:
                        minimal_facts["us-gaap"] = us_gaap
                if minimal_facts:
                    minimal_data["facts"] = minimal_facts

            prompt = f"""
            Return ONLY valid JSON with no additional text.
            Extract these financial details from SEC EDGAR data:
            - Company name
            - CIK number
            - SIC code and description
            - Recent revenue figures
            - Recent net income figures
            - Total assets
            - Total liabilities
            - Current fiscal year
            - Recent filings

            Format your response as a JSON object with these exact keys:
            {{
                "company_name": "string",
                "cik": "string",
                "sic": "string",
                "sic_description": "string",
                "revenue": "string", 
                "net_income": "string",
                "total_assets": "string",
                "total_liabilities": "string",
                "fiscal_year": "string",
                "recent_filings": [
                    {{
                        "form": "string",
                        "filingDate": "string",
                        "documentUrl": "string", 
                        "description": "string"
                    }}
                ],
                "financial_summary": "string"
            }}

            SEC EDGAR data: {json.dumps(minimal_data)}
            """

            response = self.llm.invoke(prompt)
            content = response.content

            json_match = re.search(r'({[\s\S]*})', content)

            if json_match:
                try:
                    result = json.loads(json_match.group(1))

                    filings = []
                    for filing in result.get("recent_filings", []):
                        if "description" not in filing:
                            filing["description"] = f"{filing.get('form', '')} filing"
                        filings.append(FilingModel(**filing))

                    financial_model = FinancialModel(
                        company_name=result.get("company_name", "Unknown"),
                        cik=result.get("cik", ""),
                        sic=result.get("sic", ""),
                        sic_description=result.get("sic_description", ""),
                        revenue=result.get("revenue", ""),
                        net_income=result.get("net_income", ""),
                        total_assets=result.get("total_assets", ""),
                        total_liabilities=result.get("total_liabilities", ""),
                        market_cap="",
                        fiscal_year=result.get("fiscal_year", ""),
                        recent_filings=filings,
                        financial_summary=result.get("financial_summary",
                                                     f"Financial data for {result.get('company_name', 'Unknown')}"),
                        technical={"data_source": "SEC EDGAR", "processing_method": "LLM"}
                    )

                    return financial_model.model_dump_json()
                except json.JSONDecodeError:
                    return None
            else:
                return None

        except Exception as e:
            error_details = {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "content": content if 'content' in locals() else "No content available"
            }

            return FinancialModel(
                company_name="Error",
                cik="Error",
                sic="",
                sic_description="",
                revenue="",
                net_income="",
                total_assets="",
                total_liabilities="",
                market_cap="",
                fiscal_year="",
                recent_filings=[],
                financial_summary="Error processing financial data",
                technical=error_details
            ).model_dump_json()