from rag_engine import hybrid_search
from reranker import rerank_documents

query = "What is retrieval augmented generation?"

candidates = hybrid_search(
    query,
    top_k=3
)

print("\nHYBRID CANDIDATES:")

for i, result in enumerate(candidates, start=1):
    print(
        i,
        "Page:", result["page"],
        "Hybrid:", round(result["hybrid_score"], 4)
    )

reranked_results = rerank_documents(
    query,
    candidates,
    top_k=3
)

print("\nRERANKED RESULTS:")

for i, result in enumerate(reranked_results, start=1):
    print("\n-----------------------------")
    print("RESULT", i)
    print("Page:", result["page"])
    print("Chunk:", result["chunk_id"])
    print("Rerank Score:", result["rerank_score"])
    print("Text:", result["text"][:500])