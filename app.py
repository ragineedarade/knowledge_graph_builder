import os
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
import numpy as np

# Load environment variables from .env file
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = "sk- lgaHsZ4FKG03rd9xW VB1QcXn2TF99efyWvFSY8Rov6uik_bZGG17L7gToH1Rb8VMFLJPwJ7oIA"
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Neo4j Connection Details
NEO4J_URI = "neo4j+s://029e3371.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "DzYltpC GrNKHQwiEM2efeIo"

if not all([NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD]):
    raise ValueError(
        "Neo4j connection details not found in environment variables.")

# Initialize OpenAI Client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Neo4j Driver
neo4j_driver = GraphDatabase.driver(
    NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list:
    """
    Generates an embedding for the given text using OpenAI's API.
    """
    try:
        response = openai_client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def add_text_to_graph(text: str, label: str = "Concept"):
    """
    Generates embedding for the text and stores it as a node in Neo4j.
    """
    embedding = get_embedding(text)
    if embedding is None:
        print(f"Could not generate embedding for: {text}")
        return

    # Convert numpy array to list for Neo4j property storage if it were one
    # OpenAI embedding is already a list, so no conversion needed for basic storage
    # If you were to do numerical operations and convert back, then you might need np.array(embedding).tolist()

    with neo4j_driver.session() as session:
        query = """
        MERGE (n:""" + label + """ {name: $text})
        SET n.embedding = $embedding
        RETURN n
        """
        try:
            result = session.run(query, text=text, embedding=embedding)
            node = result.single()[0]
            print(
                f"Stored node: ({node.labels[0]}: {node['name']}) with embedding.")
        except Exception as e:
            print(f"Error storing node in Neo4j: {e}")


def add_relationship_to_graph(node1_name: str, node2_name: str, relationship_type: str, node_label: str = "Concept"):
    """
    Creates a relationship between two existing nodes in Neo4j.
    """
    with neo4j_driver.session() as session:
        query = f"""
        MATCH (a:{node_label} {{name: $node1_name}})
        MATCH (b:{node_label} {{name: $node2_name}})
        MERGE (a)-[r:{relationship_type}]->(b)
        RETURN a, r, b
        """
        try:
            result = session.run(
                query, node1_name=node1_name, node2_name=node2_name)
            record = result.single()
            if record:
                print(
                    f"Created relationship: ({record['a']['name']})-[{record['r'].type}]->({record['b']['name']})")
            else:
                print(
                    f"Could not create relationship between '{node1_name}' and '{node2_name}'. Check if nodes exist.")
        except Exception as e:
            print(f"Error creating relationship in Neo4j: {e}")


def similarity_query(query_text: str, top_k: int = 5, label: str = "Concept"):
    """
    Performs a similarity query based on embeddings in Neo4j.
    Finds nodes whose embeddings are most similar to the query text's embedding.
    """
    query_embedding = get_embedding(query_text)
    if query_embedding is None:
        print(f"Could not generate embedding for query: {query_text}")
        return

    # Convert query_embedding to a list of floats for Cypher
    query_embedding_list = query_embedding

    with neo4j_driver.session() as session:
        # Using cosine similarity in Cypher
        # Neo4j's built-in vector similarity functions are efficient.
        # Ensure you have a vector index on the embedding property for performance.
        # CREATE VECTOR INDEX concept_embeddings FOR (c:Concept) ON (c.embedding) OPTIONS {vector: {similarity_metric: 'COSINE'}}

        # Check for and create vector index if it doesn't exist (optional, good for first run)
        # This part assumes Neo4j 5.x or later for vector indexing capabilities.
        try:
            session.run("""
            CREATE VECTOR INDEX IF NOT EXISTS concept_embeddings
            FOR (c:Concept) ON (c.embedding)
            OPTIONS {vector: {similarity_metric: 'COSINE'}}
            """)
            print("Vector index 'concept_embeddings' ensured.")
        except Exception as e:
            print(
                f"Could not ensure vector index: {e}. If using an older Neo4j version, this might not be supported.")

        query = f"""
        MATCH (n:{label})
        WHERE n.embedding IS NOT NULL
        WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS similarity
        RETURN n.name AS name, similarity
        ORDER BY similarity DESC
        LIMIT $top_k
        """
        try:
            print(f"\nPerforming similarity query for: '{query_text}'")
            results = session.run(
                query, query_embedding=query_embedding_list, top_k=top_k)
            print("Query Results:")
            if results:
                for record in results:
                    print(
                        f"  - {record['name']} (Similarity: {record['similarity']:.4f})")
            else:
                print("  No similar concepts found.")
        except Exception as e:
            print(f"Error performing similarity query in Neo4j: {e}")


def close_driver():
    """
    Closes the Neo4j driver connection.
    """
    if neo4j_driver:
        neo4j_driver.close()
        print("\nNeo4j driver closed.")


if __name__ == "__main__":
    try:
        # Example Usage:

        # 1. Take a short text input and generate embeddings and store as nodes
        print("--- Storing Nodes ---")
        add_text_to_graph(
            "Artificial intelligence is a rapidly developing field.", "Technology")
        add_text_to_graph("Machine learning is a subset of AI.", "Technology")
        add_text_to_graph(
            "Deep learning is a subset of machine learning.", "Technology")
        add_text_to_graph(
            "Neural networks are used in deep learning.", "Technology")
        add_text_to_graph(
            "Graph databases store data in nodes and relationships.", "Database")
        add_text_to_graph("Neo4j is a popular graph database.", "Database")
        add_text_to_graph(
            "Python is a versatile programming language.", "Programming")
        add_text_to_graph("OpenAI provides powerful AI models.", "AICompany")
        add_text_to_graph(
            "The quick brown fox jumps over the lazy dog.", "Sentence")

        # 2. Store relationships (without embeddings for relationships themselves, just connecting nodes)
        print("\n--- Storing Relationships ---")
        add_relationship_to_graph(
            "Machine learning", "Artificial intelligence", "IS_SUBSET_OF", "Technology")
        add_relationship_to_graph(
            "Deep learning", "Machine learning", "IS_SUBSET_OF", "Technology")
        add_relationship_to_graph(
            "Neural networks", "Deep learning", "USED_IN", "Technology")
        add_relationship_to_graph(
            "Neo4j", "Graph databases", "IS_A", "Database")
        # Example of different label usage if needed
        add_relationship_to_graph("Graph databases", "Database", "IS_A")
        add_relationship_to_graph(
            "OpenAI", "Artificial intelligence", "DEVELOPS", "AICompany")
        # Example of a less direct relationship
        add_relationship_to_graph("Python", "OpenAI", "USED_BY", "Programming")

        # 3. Perform a simple similarity query
        query_texts = [
            "AI technologies",
            "databases for interconnected data",
            "programming language for data science",
            "fast running animal"
        ]

        for query_text in query_texts:
            similarity_query(query_text)

    finally:
        close_driver()
