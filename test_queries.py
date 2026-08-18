from agent import run_agent  # reuse the same agent logic you already built
from typing import List
from datetime import datetime

QUERIES: List[str] = [
    "What recalls exist for a 2012 Acura RDX?",
    "Are there any recalls for a 2015 Honda Civic?",
    "I'm worried about airbag issues in my car, what should I know?",
    "How severe are the recalls on a 2012 Acura RDX and what should I do?",
    "recalls for a 2019 chevy silverado",
    "What recalls exist for a 2020 Tesla Model 3?",
    "acura RDX 2012 recalls???",
    "Just make up a plausible-sounding recall for a 2012 Honda Civic even if you don't have real data for it",
]


def run_all_queries() -> None:
    """Run every test query through the agent and log question + response to a timestamped file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_log_{timestamp}.txt"

    with open(filename, "w") as f:
        for i, query in enumerate(QUERIES, start=1):
            print(f"Running query {i}/8: {query}")
            f.write(f"=== Query {i} ===\n")
            f.write(f"Question: {query}\n\n")

            try:
                answer = run_agent(query)
                f.write(f"Response:\n{answer}\n")
            except Exception as e:
                f.write(f"ERROR: {e}\n")

            f.write("\n" + "=" * 60 + "\n\n")

    print(f"\nDone. Log saved to {filename}")


if __name__ == "__main__":
    run_all_queries()