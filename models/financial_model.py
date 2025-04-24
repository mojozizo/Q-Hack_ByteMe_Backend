from typing import Optional, List
from pydantic import BaseModel


class FilingModel(BaseModel):
    """Model for an individual SEC filing."""
    form: str
    filingDate: str
    documentUrl: Optional[str]
    description: Optional[str]


class FinancialModel(BaseModel):
    """Model for financial data from SEC EDGAR."""
    company_name: str
    cik: str
    sic: Optional[str]
    sic_description: Optional[str]
    revenue: Optional[str]
    net_income: Optional[str]
    total_assets: Optional[str]
    total_liabilities: Optional[str]
    market_cap: Optional[str]
    fiscal_year: Optional[str]
    recent_filings: Optional[List[FilingModel]]
    financial_summary: Optional[str]
    technical: Optional[dict] = None