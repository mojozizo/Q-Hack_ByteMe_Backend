from pydantic import BaseModel, Field
from typing import Optional

class Category(BaseModel):
    """
    Pydantic model for startup metrics with integer values
    """
    # Financial metrics
    annual_recurring_revenue: Optional[int] = Field(None, description="Annual Recurring Revenue (ARR)")
    monthly_recurring_revenue: Optional[int] = Field(None, description="Monthly Recurring Revenue (MRR) in USD")
    customer_acquisition_cost: Optional[int] = Field(None, description="Customer Acquisition Cost (CAC)")
    customer_lifetime_value: Optional[int] = Field(None, description="Customer Lifetime Value (CLTV)")
    cltv_cac_ratio: Optional[int] = Field(None, description="CLTV/CAC Ratio")
    gross_margin: Optional[int] = Field(None, description="Gross Margin")
    revenue_growth_rate_yoy: Optional[int] = Field(None, description="Revenue Growth Rate (YoY)")
    revenue_growth_rate_mom: Optional[int] = Field(None, description="Revenue Growth Rate (MoM)")
    
    # Operational metrics
    sales_cycle_length: Optional[int] = Field(None, description="Sales Cycle Length")
    monthly_active_users: Optional[int] = Field(None, description="Monthly Active Users (MAU)")
    user_growth_rate_yoy: Optional[int] = Field(None, description="User Growth Rate (YoY)")
    user_growth_rate_mom: Optional[int] = Field(None, description="User Growth Rate (MoM)")
    conversion_rate: Optional[int] = Field(None, description="Conversion Rate (Free to Paid)")
    
    # Strategic metrics
    pricing_strategy_maturity: Optional[int] = Field(None, description="Pricing Strategy Maturity")
    burn_rate: Optional[int] = Field(None, description="Burn Rate (monthly)")
    runway: Optional[int] = Field(None, description="Runway (in months)")
    
    # Boolean represented as int (0/1)
    ip_protection: Optional[int] = Field(None, description="IP protection of solution (yes/no)")
    
    # Scale metrics
    market_competitiveness: Optional[int] = Field(None, description="Competitiveness of the market (scale 1-5)")
    market_timing: Optional[int] = Field(None, description="Good timing -> push through regulations/trends/… (scale 1-5)")
    cap_table_cleanliness: Optional[int] = Field(None, description="Cleanliness of cap table / holding structure (scale 1-5, 5 is better)")
    
    # Founder-related metrics (represented as integers)
    founder_industry_experience: Optional[int] = Field(None, description="Founder experience in industry")
    founder_past_exits: Optional[int] = Field(None, description="Founder track record of past exits?")
    founder_background: Optional[int] = Field(None, description="Founder background at target companies/universities")
    
    # Location data (keeping as string since countries aren't meaningfully represented as integers)
    country_of_headquarters: Optional[str] = Field(None, description="Country of headquarters")