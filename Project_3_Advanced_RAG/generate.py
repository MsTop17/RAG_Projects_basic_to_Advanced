import os
from dotenv import load_dotenv
from google import genai
from rag_engine import hybrid_search
from reranker import rerank_documents

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def generate_answer(query):

    candidates = hybrid_search(
        query,
        top_k=10
    )

    reranked_results = rerank_documents(
        query,
        candidates,
        top_k=3
    )

    context = "\n\n".join(
    f"Page {result['page']}, Chunk {result['chunk_id']}:\n{result['text']}"
    for result in reranked_results
)

    prompt = f"""
You are a helpful assistant answering questions about a PDF document.

Answer the user's question using ONLY the provided context.

If the answer is not present in the context,
say that the information is not available in the provided document.

Context:
{context}

Question:
{query}

Answer:
"""

    try:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    answer = response.text

    except Exception as e:
        print(f"\nGemini generation unavailable: {e}")

        answer = (
            "Gemini is currently unavailable. "
            "The relevant document context was successfully retrieved."
        )

    return answer, reranked_results


if __name__ == "__main__":

    query = input("\nAsk a question: ")

    answer, sources = generate_answer(query)

    print("\n==============================")
    print("GENERATED ANSWER")
    print("==============================")
    print(answer)

    print("\n==============================")
    print("SOURCES")
    print("==============================")

    for result in sources:
        print(
            f"Page {result['page']} | "
            f"Chunk {result['chunk_id']} | "
            f"Rerank Score: {result['rerank_score']}"
        )

"""PDF
 ↓
ChromaDB
 ↓
Vector Search ──┐
                ├──→ Hybrid Retrieval
BM25 Search ────┘
                      ↓
                  Reranking
                      ↓
                Relevant Context
                      ↓
                    Gemini
                      ↓
                 Final Answer
"""

# import os
# from dotenv import load_dotenv
# from google import genai
# from rag_engine import hybrid_search
# from reranker import rerank_documents

# # load_dotenv()
# load_dotenv()
# client = genai.Client(
#     api_key=os.getenv("GEMINI_API_KEY")
# )
# def generate_answer(query):

#     candidates = hybrid_search(
#         query,
#         top_k=3
#     )

#     reranked_results = rerank_documents(
#         query,
#         candidates,
#         top_k=2
#     )

#     context = "\n\n".join(
#         result["text"]
#         for result in reranked_results
#     )

#     prompt = f"""
# You are a helpful assistant answering questions about a PDF document.

# Answer the user's question using ONLY the provided context.

# If the answer is not present in the context,
# say that the information is not available in the provided document.

# Context:
# {context}

# Question:
# {query}

# Answer:
# """

#     response = client.models.generate_content(
#         model="gemini-3.5-flash",
#         contents=prompt
#     )

#     return response.text


# if __name__ == "__main__":

#     query = input("\nAsk a question: ")

#     answer = generate_answer(query)

#     print("\n==============================")
#     print("GENERATED ANSWER")
#     print("==============================")
#     print(answer)

# """def generate_answer(query, results):
#     context_parts = []
#     for result in results:
#         context_parts.append(
#             f"[Page {result['page']}]\n{result['text']}"
#         )
#     context = "\n\n".join(context_parts)
#     prompt = f"""
# # You are a helpful assistant answering questions about a PDF document.
# # Answer the question using ONLY the provided context.
# # Rules:
# # 1. Do not use outside knowledge.
# # 2. If the answer cannot be found in the context, say:
# #    "The answer cannot be found in the document."
# # 3. Give a clear and concise answer.
# # 4. Mention the page numbers used.

# # Context:
# # {context}
# # Question:
# # {query}
# # Answer:
# """
#     response = client.models.generate_content(
#         model="gemini-3.5-flash",
#         contents=prompt
#     )
#     return response.text

# if __name__ == "__main__":
#     query = "What is  japan"
#     results = search_documents(
#         query,
#         top_k=3
#     )
#     answer = generate_answer(
#         query,
#         results
#     )
#     if not results:
#         print("\nANSWER:")
#         print("The answer cannot be found in the document.")
#         exit()

#     print("\nQUESTION:")
#     print(query)
#     print("\nANSWER:")
#     print(answer)
#     print("\nSOURCES:")
#     for result in results:
#         print(
#             f"Page {result['page']} | "
#             f"Distance: {result['distance']:.4f}"
#         )"""
#     """Query
#  ↓
# Gemini embedding
#  ↓
# ChromaDB vector search
#  ↓
# distance filtering
#  ↓
# Gemini
#  ↓
# answer"""

# # """                Query
# #                   ↓
# #         ┌─────────┴─────────┐
# #         ↓                   ↓
# #    Vector Search       Keyword Search
# #    (meaning)           (exact words)
# #         ↓                   ↓
# #         └─────────┬─────────┘
# #                   ↓
# #            Combined ranking
# #                   ↓
# #              Top results
# #                   ↓
#             #    Gemini"""