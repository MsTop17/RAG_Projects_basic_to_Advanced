import os
import numpy as np
import json
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

PDF_PATH = "data/Practical_Guide_to_Building_RAG.pdf"

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text:
            pages.append({
                "page": page_number,
                "text": text
            })
    return pages

def create_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk_id, chunk_text in enumerate(page_chunks):
            chunks.append({
                "chunk_id": chunk_id,
                "page": page["page"],
                "text": chunk_text
            })

    return chunks


# pages = extract_text(PDF_PATH)
# chunks = create_chunks(pages)
# print("Pages extracted:", len(pages))
# print("Total chunks:", len(chunks))
# print("\nFirst chunk:\n")
# print(chunks[0]["text"])

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values


def load_or_create_embeddings():
    if os.path.exists("embeddings.json"):
        print("Loading existing embeddings...")
        with open("embeddings.json", "r", encoding="utf-8") as f:
            return json.load(f)
    print("Creating embeddings...")
    pages = extract_text(PDF_PATH)
    chunks = create_chunks(pages)
    for chunk in chunks:
        chunk["embedding"] = get_embedding(chunk["text"])
    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    print("Embeddings created and saved.")
    return chunks

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

# def search_documents(query, top_k=3):
#     query_embedding = get_embedding(query)
#     results = []
#     for chunk in chunks:
#         score = cosine_similarity(
#             query_embedding,
#             chunk["embedding"]
#         )
#         results.append({
#             "page": chunk["page"],
#             "chunk_id": chunk["chunk_id"],
#             "text": chunk["text"],
#             "score": score
#         })
#     results.sort(
#         key=lambda x: x["score"],
#         reverse=True
#     )
#     return results[:top_k]

def search_documents(query, chunks, top_k=3):
    query_embedding = get_embedding(query)
    results = []
    for chunk in chunks:
        score = cosine_similarity(
            query_embedding,
            chunk["embedding"]
        )
        results.append({
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": score
        })
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )
    return results[:top_k]

def generate_answer(query, results):
    context_parts = []
    for result in results:
        context_parts.append(
            f"[Page {result['page']}]\n{result['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful assistant answering questions about a PDF document.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not use outside knowledge.
2. If the context does not contain enough information, say:
   "The answer cannot be found in the provided document."
3. Keep the answer clear and concise.
4. At the end, mention the page numbers used.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text

# test_embedding = get_embedding(
#     chunks[0]["text"]
# )
# print("\nEmbedding length:", len(test_embedding))
# print("First 3 values:", test_embedding[:3])



# for chunk in chunks:
#     chunk["embedding"] = get_embedding(chunk["text"])

# print("\nEmbeddings created:", len(chunks))

# with open("embeddings.json", "w", encoding="utf-8") as f:
#     json.dump(chunks, f)

# print("Embeddings saved successfully.")

# chunks = load_or_create_embeddings()
# print("Total chunks:", len(chunks))
# print("First chunk:")
# print(chunks[0]["text"])

# query = "What is retrieval augmented generation?"
# results = search_documents(query,chunks, top_k=3)
# print("\n\nSEARCH RESULTS:")
# for i, result in enumerate(results, start=1):
#     print("\n-----------------------------")
#     print("RESULT", i)
#     print("Page:", result["page"])
#     print("Chunk:", result["chunk_id"])
#     print("Score:", round(result["score"], 4))
#     print("Text:", result["text"][:500])
    
# answer = generate_answer(query, results)
# print("\n\nGENERATED ANSWER:")
# print(answer)

chunks = load_or_create_embeddings()

print("Total chunks:", len(chunks))

print("First chunk:")

print(chunks[0]["text"])


if __name__ == "__main__":

    query = "What is retrieval augmented generation?"

    results = search_documents(query,chunks, top_k=3)

    print("\nSEARCH RESULTS:")

    for i, result in enumerate(results, start=1):

        print("\n-----------------------------")
        print("RESULT", i)
        print("Page:", result["page"])
        print("Chunk:", result["chunk_id"])
        print("Score:", round(result["score"], 4))
        print("Text:", result["text"][:500])

    answer = generate_answer(query, results)

    print("\nGENERATED ANSWER:")
    print(answer)   