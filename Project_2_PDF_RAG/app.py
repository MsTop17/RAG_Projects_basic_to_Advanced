import streamlit as st

from rag import search_documents, generate_answer, load_or_create_embeddings


st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF RAG Assistant")
st.write("Ask questions about the research paper.")

# Load saved embeddings
chunks = load_or_create_embeddings()

query = st.text_input(
    "Ask a question:",
    placeholder="What is Retrieval-Augmented Generation?"
)

if st.button("Ask"):

    if query.strip():

        with st.spinner("Searching the document..."):

            # Pass chunks here
            results = search_documents(
                query,
                chunks,
                top_k=3
            )

            answer = generate_answer(
                query,
                results
            )

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")

        for i, result in enumerate(results, start=1):

            st.write(
                f"**Source {i} — Page {result['page']} "
                f"| Score: {result['score']:.3f}**"
            )

            with st.expander("View retrieved text"):
                st.write(result["text"])