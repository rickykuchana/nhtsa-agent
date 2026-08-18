import psycopg2  # connects to your PostgreSQL/RDS database
import boto3  # AWS SDK, used here to call Bedrock's embedding model
import json  # encodes/decodes the request and response bodies for Bedrock
import chromadb  # the vector database that stores and searches embeddings
from typing import List, Tuple
import 

DB_HOST = "nhtsa-recalls-db.cofgcya2a0e7.us-east-1.rds.amazonaws.com"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = os.environ.get("DB_PASSWORD")


def get_connection() -> psycopg2.extensions.connection:
    """Open a new connection to the RDS Postgres database."""
    try:
        return psycopg2.connect(
            host=DB_HOST,
            port=5432,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
    except psycopg2.OperationalError as e:
        raise RuntimeError(f"Could not connect to RDS: {e}")


def fetch_all_recalls() -> List[Tuple[str, str, str, int, str]]:
    """Pull every recall row from RDS as a list of tuples."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT campaign_number, make, model, model_year, summary FROM recalls;")
        rows = cur.fetchall()
        cur.close()
        return rows
    except psycopg2.Error as e:
        raise RuntimeError(f"Failed to fetch recalls: {e}")
    finally:
        conn.close()


def build_chunks(rows: List[Tuple[str, str, str, int, str]]) -> Tuple[List[str], List[str]]:
    """Turn each database row into one readable text chunk, paired with its campaign number as an id."""
    chunks: List[str] = []
    ids: List[str] = []
    for campaign_number, make, model, model_year, summary in rows:
        text = f"Campaign {campaign_number}: {model_year} {make} {model}. Summary: {summary}"
        chunks.append(text)
        ids.append(campaign_number)
    return chunks, ids


def embed_text(bedrock_client: "boto3.client", text: str) -> List[float]:
    """Call Bedrock's Titan embeddings model on one piece of text and return the resulting vector."""
    body = {"inputText": text}
    try:
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-embed-text-v1",
            body=json.dumps(body)
        )
        result = json.loads(response["body"].read())
        return result["embedding"]
    except Exception as e:
        raise RuntimeError(f"Failed to embed text via Bedrock: {e}")


def main() -> None:
    """Fetch recalls from RDS, embed them via Bedrock, and store them in a persistent ChromaDB collection."""
    print("Fetching recalls from RDS...")
    rows = fetch_all_recalls()
    print(f"Fetched {len(rows)} rows.")

    chunks, ids = build_chunks(rows)

    print("Connecting to Bedrock...")
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")

    print("Embedding chunks (this may take a bit)...")
    embeddings: List[List[float]] = []
    for i, chunk in enumerate(chunks):
        embedding = embed_text(bedrock_client, chunk)
        embeddings.append(embedding)
        if i % 20 == 0:
            print(f"  Embedded {i}/{len(chunks)}")

    print("Storing in ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_store")
    collection = chroma_client.get_or_create_collection(name="recalls")

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    print(f"Done. Stored {len(chunks)} recall chunks in the vector store.")


if __name__ == "__main__":
    main()