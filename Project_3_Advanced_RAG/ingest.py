import os

import chromadb
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
        page_chunks = splitter.split_text(
            page["text"]
        )
        for chunk_id, chunk_text in enumerate(page_chunks):
            chunks.append({
                "chunk_id": chunk_id,
                "page": page["page"],
                "text": chunk_text
            })
    return chunks


chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)
"""creates a persistent local vector database.
Instead of:
embeddings.json we'll eventually have:
chroma_db/
containing the vector database."""
collection = chroma_client.get_or_create_collection(
    name="research_documents"
)
"""is basically creating the collection where our document chunks will live"""

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

def build_vector_database():
    pages = extract_text(PDF_PATH)
    chunks = create_chunks(pages)
    print("Pages extracted:", len(pages))
    print("Chunks created:", len(chunks))
    for chunk in chunks:
        embedding = get_embedding(
            chunk["text"]
        )
        collection.add(
            ids=[
                str(chunk["chunk_id"]) + "_" + str(chunk["page"])
            ],
            embeddings=[embedding],
            documents=[chunk["text"]],
            metadatas=[{
                "page": chunk["page"],
                "chunk_id": chunk["chunk_id"]
            }]
        )
    print("Vector database created successfully!")

if __name__ == "__main__":
    build_vector_database()