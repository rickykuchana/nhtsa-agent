import psycopg2
import chromadb
import boto3
import json
import os
from typing import List, Tuple, Dict, Any
import anthropic
import os

DB_HOST = "nhtsa-recalls-db.cofgcya2a0e7.us-east-1.rds.amazonaws.com"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = os.environ.get("DB_PASSWORD")


def get_connection() -> psycopg2.extensions.connection:
    """Open a new connection to the RDS Postgres database."""
    return psycopg2.connect(
        host=DB_HOST,
        port=5432,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def lookup_recalls(make: str, model: str, model_year: int) -> Dict[str, Any]:
    """Query RDS for recalls matching an exact make, model, and year.
    Also checks whether the make exists anywhere in the dataset, so the agent
    can tell the difference between a confirmed empty result and a dataset
    coverage gap (the make was never fetched into the database at all)."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT campaign_number, make, model, model_year, summary FROM recalls "
        "WHERE LOWER(make) = LOWER(%s) AND LOWER(model) = LOWER(%s) AND model_year = %s;",
        (make, model, model_year)
    )
    exact_matches = cur.fetchall()

    cur.execute(
        "SELECT COUNT(*) FROM recalls WHERE LOWER(make) = LOWER(%s);",
        (make,)
    )
    make_exists_count = cur.fetchone()[0]

    cur.close()
    conn.close()

    return {
        "matches": exact_matches,
        "make_in_dataset": make_exists_count > 0
    }


def embed_query(bedrock_client: "boto3.client", text: str) -> List[float]:
    """Embed a search query using the same Bedrock model used to build the vector store."""
    body = {"inputText": text}
    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        body=json.dumps(body)
    )
    result = json.loads(response["body"].read())
    return result["embedding"]


def semantic_search(query: str, n_results: int = 3) -> List[str]:
    """Search the vector store for recall chunks semantically similar to the query."""
    bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")
    query_embedding = embed_query(bedrock_client, query)

    chroma_client = chromadb.PersistentClient(path="./chroma_store")
    collection = chroma_client.get_or_create_collection(name="recalls")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["documents"][0]


def summarise_and_recommend(recall_info: str) -> str:
    """Ask Claude to summarize recall information and recommend next steps."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"Summarize this recall information, assess severity, and recommend next steps for the vehicle owner:\n\n{recall_info}"
            }
        ]
    )
    return response.content[0].text


TOOLS = [
    {
        "name": "lookup_recalls",
        "description": (
            "Look up recalls for an exact vehicle make, model, and year directly from the database. "
            "The result includes a 'make_in_dataset' flag. If this flag is false, it means this make "
            "was never fetched into the dataset at all, and the agent should tell the user it has no "
            "data on this vehicle rather than saying no recalls exist. Only say 'no recalls found' when "
            "make_in_dataset is true and matches is empty."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "make": {"type": "string", "description": "Vehicle make, e.g. Acura"},
                "model": {"type": "string", "description": "Vehicle model, e.g. RDX"},
                "model_year": {"type": "integer", "description": "Vehicle model year, e.g. 2012"}
            },
            "required": ["make", "model", "model_year"]
        }
    },
    {
        "name": "semantic_search",
        "description": "Search recall data by meaning, useful for vague or descriptive questions that don't specify an exact make/model/year.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "summarise_and_recommend",
        "description": "Summarize retrieved recall information and recommend next steps for the vehicle owner.",
        "input_schema": {
            "type": "object",
            "properties": {
                "recall_info": {"type": "string", "description": "Raw recall text to summarize"}
            },
            "required": ["recall_info"]
        }
    }
]


def run_tool(tool_name: str, tool_input: Dict[str, Any]) -> Any:
    """Actually execute the tool Claude asked for, based on its name."""
    if tool_name == "lookup_recalls":
        return lookup_recalls(tool_input["make"], tool_input["model"], tool_input["model_year"])
    elif tool_name == "semantic_search":
        return semantic_search(tool_input["query"])
    elif tool_name == "summarise_and_recommend":
        return summarise_and_recommend(tool_input["recall_info"])
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


def run_agent(user_question: str) -> str:
    """Run the full tool-calling loop: Claude picks tools, we execute them, Claude gives a final answer."""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": user_question}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            tools=TOOLS,
            messages=messages
        )

        if response.stop_reason != "tool_use":
            return response.content[0].text

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result)
                })

        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    print("Starting agent...")
    question = "What recalls exist for a 2020 Tesla Model 3?"
    answer = run_agent(question)
    print(answer)