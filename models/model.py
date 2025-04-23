from pydantic import BaseModel, Field
from typing import List, Optional, Union

class HardFundCriteria(BaseModel):
    annual_recurring_revenue: float = Field(..., description="Annual Recurring Revenue (ARR) in USD")
    required_funding_amount: float = Field(..., description="Required Funding Amounts in USD")
    industry: List[str] = Field(..., description="Industry (Multi Select)")
    innovativeness: bool = Field(..., description="Innovativeness")
    age_of_company: float = Field(..., description="Age of company in years")

class TeamCriteria(BaseModel):
    experience_in_branches: bool = Field(..., description="Experience in Branches")
    complementary_skills: bool = Field(..., description="Complementary Skills/Background")
    ambition_motivation: float = Field(..., description="Ambition / Motivation (1-10)")
    full_time_commitment: bool = Field(..., description="Full-time commitment")
    financial_commitment: bool = Field(..., description="Financial commitment")
    resilience: bool = Field(..., description="Resilience")
    number_of_employees: int = Field(..., description="Number of employees")
    number_of_cofounders: int = Field(..., description="Number of co-founders")
    networking_score: float = Field(..., description="Networking Score (1-10)")
    publicity: float = Field(..., description="Publicity (1-10)")
    past_exits: bool = Field(..., description="Track record of past exits")
    target_companies_universities: bool = Field(..., description="Target companies/universities in CV")
    coachability: bool = Field(..., description="Coachability")

class ProblemCriteria(BaseModel):
    problem_pressing: bool = Field(..., description="Is the problem pressing and relevant?")
    willingness_to_pay: bool = Field(..., description="Willingness to pay for solution?")
    affected_people: int = Field(..., description="How many people are affected by the problem?")

class SolutionCriteria(BaseModel):
    solves_problem: bool = Field(..., description="Does the solution solve the problem?")
    technically_works: bool = Field(..., description="Does the solution work technically?")
    ip_protected: bool = Field(..., description="Is the solution IP protected?")
    product_development: bool = Field(..., description="How far is the product developed?")
    customer_accessibility: float = Field(..., description="How easily is the solution accessible (1-10)")
    technical_scalability: float = Field(..., description="How scalable is the technical solution (1-10)")
    time_to_value: int = Field(..., description="Time to Value in months")

class TractionCriteria(BaseModel):
    current_users: int = Field(..., description="How many users are using the solution")
    stickiness: bool = Field(..., description="Stickiness of the solution")
    customer_lifetime_value: float = Field(..., description="Customer Lifetime Value in USD")
    conversion_rate: float = Field(..., description="Conversion Rates in %")

class MarketCriteria(BaseModel):
    usp_score: float = Field(..., description="USP in comparison to competitors (1-10)")
    market_potential: bool = Field(..., description="Market potential / Market size")
    market_competitiveness: bool = Field(..., description="Red or blue ocean market")
    timing_score: float = Field(..., description="Timing score (1-10)")
    current_workarounds: int = Field(..., description="Current workarounds by users")

class BusinessModelCriteria(BaseModel):
    business_scalability: float = Field(..., description="Business model scalability (1-10)")
    margins: float = Field(..., description="Margins in %")
    scaling_direction: float = Field(..., description="Horizontal or vertical scaling (1-10)")
    network_effects: int = Field(..., description="Network effects in scaling")
    sales_cycle_length: bool = Field(..., description="Sales Cycle Length")
    customer_dependency: bool = Field(..., description="Risk of dependence on one customer")

class FinancialsCriteria(BaseModel):
    exit_possibilities: bool = Field(..., description="Exit-possibilities")
    clean_cap_table: bool = Field(..., description="Clean cap table / Holding structure")
    revenues: float = Field(..., description="Revenues in USD")
    annual_recurring_revenues: float = Field(..., description="Annual Recurring Revenues in USD")
    earnings: float = Field(..., description="Earnings in USD")
    runway: int = Field(..., description="Runway without insolvency in months")
    financial_plan_adequate: bool = Field(..., description="Is the financial plan adequate")
    customer_acquisition_cost: float = Field(..., description="Customer Acquisition Costs in USD")
    user_growth_rate: float = Field(..., description="Growth Rate (users) in %")
    revenue_growth_rate: float = Field(..., description="Growth Rate (revenues) in USD")
    monthly_cash_burn: float = Field(..., description="Cash Burn per Month in USD")
    burn_multiple: float = Field(..., description="Burn Multiple in USD")
    pending_cash_needs: float = Field(..., description="Pending short-term cash needs in USD")

class RoadmapCriteria(BaseModel):
    next_steps_adequate: bool = Field(..., description="Are the next steps adequate?")
    hiring_plan_aligned: bool = Field(..., description="Does the hiring plan align with business goals?")

class RisksCriteria(BaseModel):
    regulatory_risks: bool = Field(..., description="Regulatory Risks")
    trend_risks: bool = Field(..., description="Trend Risks")
    esg_risks: bool = Field(..., description="ESG Risks")
    shitstorm_risks: bool = Field(..., description="Shitstorm risks")
    external_dependencies: bool = Field(..., description="Dependency on External APIs or Tech Platforms")
    litigation_risks: bool = Field(..., description="Litigation or IP disputes")
    sanctions_check: bool = Field(..., description="Sanctions check")
    inflated_metrics: bool = Field(..., description="Inflated metrics / vanity metrics")
    unrealistic_growth: bool = Field(..., description="Unrealistic growth assumptions")

class RedFlagsCriteria(BaseModel):
    no_go_industries: bool = Field(..., description="Potential No Go industries")
    harmful_business_models: bool = Field(..., description="Harmful Business Models")
    founder_scandals: bool = Field(..., description="Scandals around founders / the startup")

class Category(BaseModel):
    hard_fund_criteria: HardFundCriteria
    team: TeamCriteria
    problem: ProblemCriteria
    solution: SolutionCriteria
    traction: TractionCriteria
    market: MarketCriteria
    business_model: BusinessModelCriteria
    financials: FinancialsCriteria
    roadmap: RoadmapCriteria
    risks: RisksCriteria
    red_flags: RedFlagsCriteria

detail_email = Email.model_json_schema()