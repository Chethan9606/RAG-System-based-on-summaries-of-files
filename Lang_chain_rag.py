#cd steamlit
# streamlit run ex.py
import streamlit as st
import fitz
import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

SUMMARIES_FILE = "summaries.json"
DB_DIR = "pdf_dbs"

st.title("PDF Q&A")

groq_api_key = st.secrets["GROQ_API_KEY"]

# ── Shared resources ─────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

def get_llm(api_key):
    return ChatGroq(api_key=api_key, model="llama-3.3-70b-versatile")

# ── Summaries store (JSON file) ──────────────────────────────────
def load_summaries():
    if os.path.exists(SUMMARIES_FILE):
        return json.load(open(SUMMARIES_FILE))
    return {}  # {filename: summary}

def save_summaries(summaries):
    json.dump(summaries, open(SUMMARIES_FILE, "w"), indent=2)

# ── Per-file Chroma DB ───────────────────────────────────────────
def get_chroma(filename, chunks=None):
    path = os.path.join(DB_DIR, filename.replace(".pdf", ""))
    embeddings = get_embeddings()
    if chunks:
        return Chroma.from_texts(texts=chunks, embedding=embeddings, persist_directory=path)
    return Chroma(persist_directory=path, embedding_function=embeddings)

# ── Tab 1: Index PDFs ────────────────────────────────────────────
tab1, tab2 = st.tabs(["📥 Index PDFs", "🔍 Ask a Question"])

with tab1:
    uploaded_files = st.file_uploader("Upload up to 10 PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Index PDFs") and uploaded_files and groq_api_key:
        summaries = load_summaries()
        llm = get_llm(groq_api_key)

        for uploaded_file in uploaded_files:
            fname = uploaded_file.name

            if fname in summaries:
                st.info(f"⏭ {fname} already indexed, skipping.")
                continue

            with st.spinner(f"Indexing {fname}..."):
                # Extract text
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                text = "".join(page.get_text() for page in doc)

                # Summarise with Groq
                summary = llm.invoke(
                    f"Summarise this document in 3-5 sentences for routing purposes:\n\n{text[:3000]}"
                ).content
                summaries[fname] = summary

                # Chunk + store in per-file Chroma DB
                chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_text(text)
                get_chroma(fname, chunks)

                st.success(f"✅ {fname} — {len(chunks)} chunks")
                st.caption(f"**Summary:** {summary}")

        save_summaries(summaries)
        st.success("All done! Summaries saved.")

# ── Tab 2: Ask ───────────────────────────────────────────────────
with tab2:
    summaries = load_summaries()

    if not summaries:
        st.info("Index some PDFs first.")
    else:
        st.write(f"**{len(summaries)} PDFs indexed:** {', '.join(summaries.keys())}")
        query = st.text_input("Ask a question")

        if st.button("Ask") and query and groq_api_key:
            with st.spinner("Finding relevant PDF..."):
                llm = get_llm(groq_api_key)

                # Step 1: pick the best file using summaries
                summary_block = "\n\n".join(
                    f"File: {fname}\nSummary: {s}" for fname, s in summaries.items()
                )
                routing_prompt = f"""Given these PDF summaries, which ONE file is most relevant to the question?
Reply with ONLY the exact filename (e.g. notes.pdf), nothing else.

{summary_block}

Question: {query}"""

                best_file = llm.invoke(routing_prompt).content.strip()
                st.caption(f"📂 Routing to: **{best_file}**")

                # Step 2: load that file's Chroma + build hybrid retriever
                vectorstore = get_chroma(best_file)
                bm25_chunks = vectorstore.get()["documents"] or []


                chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

                if bm25_chunks:
                    bm25 = BM25Retriever.from_texts(bm25_chunks)
                    bm25.k = 5
                    retriever = EnsembleRetriever(retrievers=[bm25, chroma_retriever], weights=[0.5, 0.5])
                else:
                    retriever = chroma_retriever

                # Step 3: answer
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "Answer using only the context. Use tables, bullet points where helpful."),
                    ("human", "Context:\n{context}\n\nQuestion: {question}")
                ])
                chain = prompt | llm | StrOutputParser()

                docs = retriever.invoke(query)
                context = "\n\n".join(d.page_content for d in docs)
                st.markdown(chain.invoke({"context": context, "question": query}))
