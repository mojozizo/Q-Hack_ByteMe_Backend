# ByteMe - Startup Analysis Platform

ByteMe is a comprehensive startup analysis platform that serves as an alternative to ACE. The platform extracts structured data from pitch decks and websites, evaluates startups through cross-referencing information with public sources, and provides valuation metrics for investment decisions.

## Problem Statement

Traditional startup evaluation is time-consuming, inconsistent, and prone to bias. ByteMe addresses these challenges by automating the extraction of critical information from pitch decks and other data sources, evaluating this information against public data, and providing structured analyses for informed investment decisions.

The platform extracts structured data (e.g., founder background, funding history, tech stack), assesses each startup's potential using ML and NLP techniques, and presents the discovered insights in a digestible format.

## System Architecture

ByteMe follows an ETL (Extract, Transform, Load) architecture:

### Extract Phase
- **Modular Extraction System** that processes pitch decks and other documents
- **Multi-Agent Framework** that coordinates specialized agents for different data sources
- **Data Sources**: PDF pitch decks, websites, LinkedIn profiles, financial data, news articles

### Transform Phase
- Data enrichment through web searches
- Structured data parsing to predefined models
- Risk assessment based on news and regulatory information

### Load Phase
- API endpoints that provide the processed data in a consistent JSON format
- Integration capabilities with frontend systems

## Key Components

### Extraction Agents
- **PDF Agent**: Extracts structured data from pitch decks
- **Web Search Agent**: Enhances extracted data with web search results
- **LinkedIn Agent**: Analyzes founder profiles and experience
- **News Agent**: Gathers recent news about the company and performs sentiment analysis
- **Financial Agent**: Retrieves financial metrics and performance data
- **Orchestrator Agent**: Coordinates all agents and consolidates results

### Data Models
- `StartupMetrics`: Comprehensive model for startup information including:
  - Company information (name, founding year, headquarters, etc.)
  - Financial metrics (ARR, MRR, CAC, CLTV, etc.)
  - Operational metrics (sales cycle, MAU, conversion rates, etc.)
  - Strategic metrics (burn rate, runway, etc.)
  - Founder metrics (experience, background, skills, etc.)
  - Risk assessment (regulatory risks, trend risks, news sentiment, etc.)

### API Endpoints
- `/upload-pdf/`: Process and analyze pitch decks
- `/news/{company_name}`: Retrieve news data for a specific company
- `/linkedin/{profile_url}`: Analyze LinkedIn profiles
- `/orchestrate/`: Run a comprehensive analysis using all agents

## Workflows

### Extraction Workflow
1. User uploads a pitch deck PDF
2. System extracts text and structured data from the document
3. Extracted data is enhanced with web search results
4. The system identifies the company name and other key information

### Evaluation Workflow
1. Using the company name from extraction, the system:
   - Retrieves financial data
   - Analyzes founder's LinkedIn profile 
   - Gathers news about the company
   - Checks for regulatory and trend risks
   - Evaluates if Politically Exposed Persons (PEPs) are involved
   - Checks for legal cases or outstanding fines

### Valuation Workflow
1. Financial metrics are evaluated and scored
2. Founder experience and track record are assessed
3. Market conditions and competitive landscape are analyzed
4. Risk factors are incorporated into the valuation
5. A comprehensive investment recommendation is generated

## Getting Started

### Build
```
docker build -t q-backend .
```

### Run
```
docker run -v $(pwd)/:/app/ -p 8000:8000 q-backend
```

## Technology Stack

- **Framework**: FastAPI
- **Language**: Python
- **NLP/ML**: LangChain, OpenAI models (GPT-4)
- **Document Processing**: PyPDF2, Document Parsers
- **Web Scraping**: Custom web search utilities
- **Data Models**: Pydantic
- **Containerization**: Docker

## Future Enhancements

- Integration with more data sources (CrunchBase, PitchBook, etc.)
- Advanced financial modeling and valuation
- Founder network analysis
- Market trend prediction
- Competitive landscape mapping
- Customizable scoring criteria based on investment thesis

## Architecture Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│  Input Sources  │     │  Extraction ETL  │     │   Output Formats   │
├─────────────────┤     ├──────────────────┤     ├────────────────────┤
│ - Pitch Decks   │     │ - PDF Agent      │     │ - Structured JSON  │
│ - LinkedIn      │────▶│ - LinkedIn Agent │────▶│ - Risk Assessment  │
│ - News Sources  │     │ - News Agent     │     │ - Metrics & Scores │
│ - Web Data      │     │ - Financial Agent│     │ - Analysis Summary │
└─────────────────┘     └──────────────────┘     └────────────────────┘
                                │
                                ▼
                        ┌──────────────────┐
                        │  Orchestration   │
                        ├──────────────────┤
                        │ Coordinated      │
                        │ multi-agent      │
                        │ workflow         │
                        └──────────────────┘
```

## Contributing

This project is part of Q-Hack by ByteMe team. Contact the contributors for more information on how to contribute.