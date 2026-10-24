"""Hybrid Search
     ↓
Top 10 candidates
     ↓
Gemini evaluates relevance
     ↓
Relevance scores
     ↓
Sort
     ↓
Top 3 chunks"""

"""Vector search — ChromaDB finds semantically similar chunks using embeddings.
Keyword search — BM25 finds chunks containing important exact words.
Hybrid search — You combine both approaches instead of relying on only one.
Reranking — Gemini gets the retrieved candidates and evaluates their relevance to the actual question.
Final retrieval pipeline — We now have the foundation of:
PDF → Chunks → Vector DB → Hybrid Retrieval → Reranking → Generation"""

import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def rerank_documents(query, documents, top_k=3):

    scored_documents = []

    for document in documents:

        prompt = f"""
Rate how relevant the following document passage is
to the user's question.

Give ONLY a number from 0 to 10.

Question:
{query}

Passage:
{document["text"]}
"""

        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )

            score = float(response.text.strip())

        except Exception as e:
            print("Gemini reranker unavailable:", e)
            score = document.get("hybrid_score", 0)

        document_copy = document.copy()
        document_copy["rerank_score"] = score

        scored_documents.append(document_copy)

    scored_documents.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    return scored_documents[:top_k]


# def rerank_documents(query, documents, top_k=3):

#     scored_documents = []

#     for document in documents:

#         prompt = f"""
# Rate how relevant the following document passage is
# to the user's question.

# Give ONLY a number from 0 to 10.

# Question:
# {query}

# Passage:
# {document["text"]}
# """

#         # response = client.models.generate_content(
#         #     model="gemini-3.5-flash",
#         #     contents=prompt
#         # )

#         # try:
#         #     score = float(response.text.strip())
#         # except ValueError:
#         #     score = 0
#         try:
#             response = client.models.generate_content(
#                 model="gemini-3.5-flash",
#                 contents=prompt
#             )

#             score = float(response.text.strip())

#         except Exception as e:
#             print("Gemini reranker unavailable:", e)
#             score = document.get("hybrid_score", 0)

#         document_copy = document.copy()
#         document_copy["rerank_score"] = score

#         scored_documents.append(document_copy)

#     scored_documents.sort(
#         key=lambda x: x["rerank_score"],
#         reverse=True
#     )

#     return scored_documents[:top_k]