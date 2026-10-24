import streamlit as st
from rag_engine import hybrid_search
from reranker import rerank_documents

st.set_page_config(
    page_title="Advanced RAG",
    page_icon="🚀",
    layout="wide"
)
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0;
}
.subtitle {
    font-size: 17px;
    color: #777;
    margin-bottom: 25px;
}
.card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
    min-height: 180px;
}
.score {
    font-size: 25px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">🚀 Advanced RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Research Intelligence • Hybrid Retrieval • Reranking • Grounded Generation'
    '</div>',
    unsafe_allow_html=True
)

query = st.text_input(
    "Ask a question about the research document",
    placeholder="Example: What is Retrieval-Augmented Generation?"
)

if st.button("🔍 Run RAG", type="primary") and query:

    with st.spinner("Running hybrid retrieval..."):

        hybrid_results = hybrid_search(
            query,
            top_k=10
        )
    st.write("QUERY:", query)

    st.write("HYBRID CANDIDATES")

    for result in hybrid_results:
        st.write(
            result["page"],
            result["chunk_id"],
            result["hybrid_score"]
        )   

    with st.spinner("Reranking results..."):

        reranked_results = rerank_documents(
            query,
            hybrid_results,
            top_k=3
        )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔎 Vector Search")

        st.markdown(
            """
            <div class="card">
            <b>Semantic Retrieval</b>
            <br><br>
            Finds conceptually similar chunks using embeddings.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("### 🔤 BM25 Search")

        st.markdown(
            """
            <div class="card">
            <b>Keyword Retrieval</b>
            <br><br>
            Finds chunks containing important query terms.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown("### 🔗 Hybrid Search")

        st.markdown(
            """
            <div class="card">
            <b>Combined Retrieval</b>
            <br><br>
            Combines semantic and keyword relevance.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.divider()

    st.markdown("### 🎯 Top Reranked Context")

    if reranked_results:

        source_cols = st.columns(len(reranked_results))

        for i, result in enumerate(reranked_results):

            with source_cols[i]:

                st.markdown(
                    f"**Page {result['page']} • Chunk {result['chunk_id']}**"
                )

                st.metric(
                    "Rerank Score",
                    round(result["rerank_score"], 4)
                )

                st.metric(
                    "Hybrid Score",
                    round(result["hybrid_score"], 4)
                )

                with st.expander("View context"):
                    st.write(result["text"])

    else:

        st.warning(
            "No reranked results were returned."
        )

    st.divider()

    st.markdown("### 🧠 Retrieval Pipeline")

    st.markdown(
        """
        **Query**
        → **Vector Search**
        → **BM25**
        → **Hybrid Search**
        → **Reranking**
        → **Top 3 Context**
        """
    )

    st.success(
        "Hybrid retrieval and reranking completed successfully."
    )

    # st.divider()

    # st.markdown("### 📚 Top Retrieved Context")

    # source_cols = st.columns(len(results))

    # for i, result in enumerate(results):

    #     with source_cols[i]:

    #         st.markdown(
    #             f"**Page {result['page']} • Chunk {result['chunk_id']}**"
    #         )

    #         st.metric(
    #             "Hybrid Score",
    #             round(result["hybrid_score"], 4)
    #         )

    #         with st.expander("View chunk"):
    #             st.write(result["text"])

    # st.divider()

    # st.markdown("### 🧠 Retrieval Pipeline")

    # st.markdown(
    #     """
    #     **Query**
    #     → **Vector Search**
    #     → **BM25**
    #     → **Hybrid Search**
    #     → **Top Context**
    #     """
    # )

    # st.success(
    #     "Hybrid retrieval completed successfully."
    # )


"""
    Question
   ↓
hybrid_search()
   ↓
10 candidates
   ↓
rerank_documents()
   ↓
3 best results
   ↓
Streamlit displays them"""