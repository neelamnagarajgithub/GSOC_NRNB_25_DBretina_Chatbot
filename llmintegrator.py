import logging
from langchain_neo4j import Neo4jGraph
from langchain_ollama import OllamaLLM  # Updated import
from langchain_neo4j.chains.graph_qa.cypher import GraphCypherQAChain
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Neo4j environment variables
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USERNAME"] = "nagaraj"
os.environ["NEO4J_PASSWORD"] = "linux9977"

# Connect to Neo4j
graph = Neo4jGraph()

# Import movie information
movies_query = """
LOAD CSV WITH HEADERS FROM 
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
AS row
MERGE (m:Movie {id:row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
FOREACH (director in split(row.director, '|') | 
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors, '|') | 
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') | 
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))
"""

graph.query(movies_query)
graph.refresh_schema()
logger.info("Graph schema: %s", graph.schema)

# Initialize Ollama LLM
llm = OllamaLLM(model="llama3.2:1b", temperature=0)

# Create GraphCypherQAChain
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True)

def execute_query(query: str):
    result = chain.invoke({"query": query})
    
    # Extract the generated Cypher query and full context if available
    generated_cypher = getattr(chain, 'generated_cypher', 'N/A')
    full_context = getattr(chain, 'full_context', 'N/A')
    
    # Log the generated Cypher query and full context
    logger.info("Generated Cypher: %s", generated_cypher)
    logger.info("Full Context: %s", full_context)
    
    # Create the response with additional details
    response = {
        "query": query,
        "result": result,
        "generated_cypher": generated_cypher,
        "full_context": full_context
    }
    
    return response