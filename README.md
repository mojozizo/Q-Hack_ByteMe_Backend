# Q-Hack_ByteMe_Backend
A ACE alternative project

### Problem:
The platform should extract structured data (e.g., founder background, funding history, tech stack), assess each startup’s potential using ML and NLP techniques (such as LLMs), and present the discovered insights

Bonus points for stretch goals like founder track record analysis, real-time demo features, or LLM-powered chat interfaces.

### Workflows:
- Extraction - Extract information from pitch decks and websites
- Evaluation - Cross check the given information within the deck with public websites, social media sites which include checks if a Politically Exposed Person (PEP) is involved or if the individual has any registered legal cases, outstanding fines etc.
- Valuation - Provide scores to the teams based on the evaluation

### Goals:
- Figure out what points do we need to extract from the pdf or do we need the 
???entire pdf in the context window with a chat based application ???

- Figure out the evaluation criterias as well as the workflows within the evaluation pipeline

- Figure out the scoring criteria

# Build
docker build -t q-backend .

# Run
docker run -v $(pwd)/:/app/ q-backend