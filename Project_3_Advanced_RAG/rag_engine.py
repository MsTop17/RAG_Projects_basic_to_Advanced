import os
import numpy as np
import chromadb
from dotenv import load_dotenv
from google import genai
from rank_bm25 import BM25Okapi

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# Connect to the existing ChromaDB database
chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_collection(
    name="research_documents"
)

all_documents = collection.get()

documents = all_documents["documents"]
metadatas = all_documents["metadatas"]

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return result.embeddings[0].values
def hybrid_search(query, top_k=3, max_distance=0.6):
    query_embedding = get_embedding(query)

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )

    vector_documents = vector_results["documents"][0]
    vector_metadatas = vector_results["metadatas"][0]
    vector_distances = vector_results["distances"][0]

    vector_candidates = {}

    for document, metadata, distance in zip(
        vector_documents,
        vector_metadatas,
        vector_distances
    ):
        if distance <= max_distance:
            key = (
                metadata["chunk_id"],
                metadata["page"]
            )

            vector_candidates[key] = {
                "page": metadata["page"],
                "chunk_id": metadata["chunk_id"],
                "text": document,
                "distance": distance
            }

    bm25_scores = bm25.get_scores(
        query.lower().split()
    )

    ranked_indices = np.argsort(
        bm25_scores
    )[::-1][:10]

    keyword_candidates = {}

    for index in ranked_indices:
        key = (
            metadatas[index]["chunk_id"],
            metadatas[index]["page"]
        )

        keyword_candidates[key] = {
            "page": metadatas[index]["page"],
            "chunk_id": metadatas[index]["chunk_id"],
            "text": documents[index],
            "keyword_score": float(
                bm25_scores[index]
            )
        }

    all_keys = set(vector_candidates) | set(keyword_candidates)

    results = []

    vector_distances_list = [
        item["distance"]
        for item in vector_candidates.values()
    ]

    keyword_scores_list = [
        item["keyword_score"]
        for item in keyword_candidates.values()
    ]

    max_keyword = max(
        keyword_scores_list
    ) if keyword_scores_list else 1

    min_distance = min(
        vector_distances_list
    ) if vector_distances_list else 1

    max_distance_value = max(
        vector_distances_list
    ) if vector_distances_list else 1

    for key in all_keys:

        vector_score = 0
        keyword_score = 0

        if key in vector_candidates:

            distance = vector_candidates[key]["distance"]

            if max_distance_value != min_distance:
                vector_score = (
                    max_distance_value - distance
                ) / (
                    max_distance_value - min_distance
                )
            else:
                vector_score = 1

        if key in keyword_candidates:

            keyword_score = (
                keyword_candidates[key]["keyword_score"]
                / max_keyword
            )

        if key in vector_candidates:
            base = vector_candidates[key]

        else:
            base = keyword_candidates[key]

        hybrid_score = (
            0.7 * vector_score
            + 0.3 * keyword_score
        )

        results.append({
            "page": base["page"],
            "chunk_id": base["chunk_id"],
            "text": base["text"],
            "vector_score": vector_score,
            "keyword_score": keyword_score,
            "hybrid_score": hybrid_score
        })

    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return results[:top_k]

"""def hybrid_search(query, top_k=3, max_distance=0.6):
    query_embedding = get_embedding(query)

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    vector_documents = vector_results["documents"][0]
    vector_metadatas = vector_results["metadatas"][0]
    vector_distances = vector_results["distances"][0]

    vector_scores = {}

    for document, metadata, distance in zip(
        vector_documents,
        vector_metadatas,
        vector_distances
    ):
        if distance <= max_distance:
            key = metadata["chunk_id"], metadata["page"]

            vector_scores[key] = {
                "page": metadata["page"],
                "chunk_id": metadata["chunk_id"],
                "text": document,
                "distance": distance,
                "vector_score": 1 / (1 + distance)
            }

    query_tokens = query.lower().split()

    bm25_scores = bm25.get_scores(query_tokens)

    ranked_indices = np.argsort(bm25_scores)[::-1][:top_k]

    hybrid_results = {}

    for key, result in vector_scores.items():
        hybrid_results[key] = result

    for index in ranked_indices:
        page = metadatas[index]["page"]
        chunk_id = metadatas[index]["chunk_id"]
        key = chunk_id, page

        keyword_score = float(bm25_scores[index])

        if key in hybrid_results:
            hybrid_results[key]["keyword_score"] = keyword_score
        else:
            hybrid_results[key] = {
                "page": page,
                "chunk_id": chunk_id,
                "text": documents[index],
                "distance": None,
                "vector_score": 0,
                "keyword_score": keyword_score
            }

    for result in hybrid_results.values():
        keyword_score = result.get("keyword_score", 0)

        result["hybrid_score"] = (
            0.7 * result["vector_score"]
            + 0.3 * keyword_score
        )

    results = list(hybrid_results.values())

    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return results[:top_k]"""
"""
def search_documents(query, top_k=3, max_distance=0.6):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    retrieved_results = []
    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):
        if distance <= max_distance:
            retrieved_results.append({
                "page": metadata["page"],
                "chunk_id": metadata["chunk_id"],
                "text": document,
                "distance": distance
            })
    return retrieved_results

def keyword_search(query, top_k=3):
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for index in ranked_indices:
        results.append({
            "page": metadatas[index]["page"],
            "chunk_id": metadatas[index]["chunk_id"],
            "text": documents[index],
            "keyword_score": float(scores[index])
        })
    return results
"""
if __name__ == "__main__":

    query = "What is retrieval augmented generation?"

    results = hybrid_search(
        query,
        top_k=3
    )

    print("\nHYBRID SEARCH RESULTS:")

    for i, result in enumerate(results, start=1):

        print("\n-----------------------------")
        print("RESULT", i)
        print("Page:", result["page"])
        print("Chunk:", result["chunk_id"])
        # print("Distance:", result["distance"])
        print("Vector Score:", round(result["vector_score"], 4))
        print("Keyword Score:", round(result.get("keyword_score", 0), 4))
        print("Hybrid Score:", round(result["hybrid_score"], 4))
        print("Text:", result["text"][:500])



"""                  QUERY
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
    ChromaDB Vector        BM25
       Search           Keyword Search
          │                   │
          ↓                   ↓
    Semantic score       Keyword score
          │                   │
          └─────────┬─────────┘
                    ↓
              NORMALIZATION
                    ↓
             HYBRID SCORE
                    ↓
               Top 3 chunks
                    ↓
                 Gemini"""


"""
ChromaDB uses embeddings.

Question
   ↓
Embedding
   ↓
ChromaDB
   ↓
Semantically similar chunks"""


"""BM25 looks for important matching words.

Question
   ↓
Words: retrieval, augmented, generation
   ↓
BM25
   ↓
Chunks containing those terms"""

"""Hybrid search combines them
                 QUERY
                   │
          ┌────────┴────────┐
          ↓                 ↓
     Vector Search      BM25 Search
       ChromaDB           Keywords
          ↓                 ↓
     Vector Score      Keyword Score
          └────────┬────────┘
                   ↓
             Hybrid Score
                   ↓
             Best chunks"""

"""                 USER QUESTION
                       │
                       ▼
              get_embedding()
                       │
                       ▼
                 CHROMADB
                       │
             semantic results
                       │
                       ▼
             distance filtering
                       │
                       ▼
               vector score
                       │
                       │
                       ├──────────────┐
                       │              │
                       ▼              ▼
                     BM25        keyword score
                       │              │
                       └──────┬───────┘
                              ▼
                       MATCH CHUNKS
                              │
                              ▼
                     HYBRID SCORING
                              │
                     70% vector
                     30% keyword
                              │
                              ▼
                        SORT RESULTS
                              │
                              ▼
                           TOP 3"""