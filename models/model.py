from pydantic import BaseModel, Field
from typing import Optional

class StartupMetrics(BaseModel):
    """
    Pydantic model for comprehensive startup metrics and information
    """
    # Company information
    company_name: Optional[str] = Field(None, description="Company Name")
    official_company_name: Optional[str] = Field(None, description="Official Company Name")
    year_of_founding: Optional[int] = Field(None, description="Year of Founding")
    location_of_headquarters: Optional[str] = Field(None, description="Location of Headquarters")
    business_model: Optional[str] = Field(None, description="Business Model")
    industry: Optional[str] = Field(None, description="Industry")
    required_funding_amount: Optional[int] = Field(None, description="Required Funding Amount")
    employees: Optional[str] = Field(None, description="Number of Employees (range)")
    website_link: Optional[str] = Field(None, description="Website Link")
    founders: Optional[str] = Field(None, description="Name and Designation of Founders")
    one_sentence_pitch: Optional[str] = Field(None, description="One Sentence Pitch")
    linkedin_profile_ceo: Optional[str] = Field(None, description="LinkedIn Profile of CEO")
    pitch_deck_summary: Optional[str] = Field(None, description="Summary of pitch deck highlighting important aspects. Minimum 3-4 lines")
    country_of_headquarters: Optional[str] = Field(None, description="Country of headquarters")
    
    # Financial metrics
    annual_recurring_revenue: Optional[int] = Field(None, description="Annual Recurring Revenue (ARR)")
    monthly_recurring_revenue: Optional[int] = Field(None, description="Monthly Recurring Revenue (MRR) in USD")
    customer_acquisition_cost: Optional[int] = Field(None, description="Customer Acquisition Cost (CAC)")
    customer_lifetime_value: Optional[int] = Field(None, description="Customer Lifetime Value (CLTV)")
    cltv_cac_ratio: Optional[int] = Field(None, description="CLTV/CAC Ratio")
    gross_margin: Optional[int] = Field(None, description="Gross Margin")
    revenue_growth_rate_yoy: Optional[int] = Field(None, description="Revenue Growth Rate (YoY)")
    revenue_growth_rate_mom: Optional[int] = Field(None, description="Revenue Growth Rate (MoM)")
    churn_rate: Optional[int] = Field(None, description="Churn Rate (users, monthly)")
    net_revenue_retention: Optional[int] = Field(None, description="Net Revenue Retention (NRR)")
    customer_payback_period: Optional[int] = Field(None, description="Customer Payback Period")
    burn_multiple: Optional[int] = Field(None, description="Burn Multiple (cash burn/net new revenue)")
    revenue_per_fte: Optional[int] = Field(None, description="Revenue per FTE")
    valuation_arr_multiple: Optional[int] = Field(None, description="Valuation / ARR Multiple")
    top_3_revenue_share: Optional[int] = Field(None, description="Top-3 Revenue Share")
    market_coverage: Optional[int] = Field(None, description="Market Coverage (Revenue / SAM)")
    
    # Operational metrics
    sales_cycle_length: Optional[int] = Field(None, description="Sales Cycle Length")
    monthly_active_users: Optional[int] = Field(None, description="Monthly Active Users (MAU)")
    user_growth_rate_yoy: Optional[int] = Field(None, description="User Growth Rate (YoY)")
    user_growth_rate_mom: Optional[int] = Field(None, description="User Growth Rate (MoM)")
    conversion_rate: Optional[int] = Field(None, description="Conversion Rate (Free to Paid)")
    dau_mau_ratio: Optional[int] = Field(None, description="DAU / MAU Ratio")
    product_stickiness: Optional[int] = Field(None, description="Product Stickiness")
    time_to_value: Optional[int] = Field(None, description="Time to Value (TTV)")
    employee_count: Optional[int] = Field(None, description="Number of employees")
    
    # Strategic metrics
    pricing_strategy_maturity: Optional[int] = Field(None, description="Pricing Strategy Maturity")
    burn_rate: Optional[int] = Field(None, description="Burn Rate (monthly)")
    runway: Optional[int] = Field(None, description="Runway (in months)")
    
    # Founder-related metrics
    founder_industry_experience: Optional[int] = Field(None, description="Founder experience in industry")
    founder_past_exits: Optional[int] = Field(None, description="Founder track record of past exits?")
    founder_background: Optional[int] = Field(None, description="Founder background at target companies/universities")
    founder_sanction_free: Optional[bool] = Field(None, description="Founder is sanction free (yes/no)")
    
    # Scale metrics (1-5)
    market_competitiveness: Optional[int] = Field(None, description="Competitiveness of the market (scale 1-5)")
    market_timing: Optional[int] = Field(None, description="Good timing -> push through regulations/trends/… (scale 1-5)")
    cap_table_cleanliness: Optional[int] = Field(None, description="Cleanliness of cap table / holding structure (scale 1-5, 5 is better)")
    business_model_scalability: Optional[int] = Field(None, description="Scalability of the business model and sales process (scale 1-5)")
    hiring_plan_alignment: Optional[int] = Field(None, description="Alignment of hiring plan and business goals (scale 1-5)")
    
    # Boolean fields
    ip_protection: Optional[bool] = Field(None, description="IP protection of solution (yes/no)")
    regulatory_risks: Optional[bool] = Field(None, description="Regulatory Risks (yes/no)")
    trend_risks: Optional[bool] = Field(None, description="Trend Risks (yes/no)")
    litigation_ip_disputes: Optional[bool] = Field(None, description="Litigation or IP disputes (yes/no)")
    company_sanction_free: Optional[bool] = Field(None, description="Company is sanction free (yes/no)")

# For backward compatibility
CompanyInfo = StartupMetrics
Category = StartupMetrics
CategoryToSearch = StartupMetrics

