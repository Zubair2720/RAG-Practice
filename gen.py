import os
import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv, find_dotenv
from google import genai
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------
# Setup
# ---------------------------------------------------------------
load_dotenv(find_dotenv(), override=True)

st.set_page_config(page_title="Gemini RAG App", page_icon="🤖", layout="wide")

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = "gemini-3.5-flash-lite"

if not API_KEY:
    st.error("GEMINI_API_KEY nahi mili. Apni .env file mein GEMINI_API_KEY set karein.")
    st.stop()

# langchain-google-genai env var se key uthata hai
os.environ["GOOGLE_API_KEY"] = API_KEY


@st.cache_resource
def get_client():
    return genai.Client(api_key=API_KEY)


@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


client = get_client()

RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the question using only the following context.
If the answer is not available in the context, say:
"I could not find this information in the document."

Context:
{context}

Question:
{question}

Answer:
"""
)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def build_vector_store(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int):
    """PDF load -> split -> embed -> Chroma (in-memory)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        docs = PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = splitter.split_documents(docs)

    collection = "docs_" + hashlib.md5(pdf_bytes).hexdigest()[:12]
    store = Chroma(
        collection_name=collection,
        embedding_function=get_embeddings(),
    )
    store.add_documents(documents=texts)
    return store, len(docs), len(texts)


def answer_question(store, query: str, k: int):
    results = store.similarity_search(query, k=k)
    context = "\n\n".join(doc.page_content for doc in results)
    prompt = RAG_PROMPT.invoke({"context": context, "question": query}).to_string()
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text, results


# ---------------------------------------------------------------
# Session state
# ---------------------------------------------------------------
st.session_state.setdefault("vector_store", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("doc_info", None)

# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    chunk_size = st.slider("Chunk size", 200, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 500, 150, step=10)
    top_k = st.slider("Top-K results (k)", 1, 10, 3)

    st.divider()
    st.subheader("📄 PDF Upload")
    pdf_file = st.file_uploader("PDF select karein", type=["pdf"])

    if st.button("Process PDF", type="primary", use_container_width=True):
        if pdf_file is None:
            st.warning("Pehle PDF upload karein.")
        else:
            with st.spinner("PDF process ho rahi hai..."):
                try:
                    store, n_pages, n_chunks = build_vector_store(
                        pdf_file.getvalue(), chunk_size, chunk_overlap
                    )
                    st.session_state.vector_store = store
                    st.session_state.doc_info = (pdf_file.name, n_pages, n_chunks)
                    st.session_state.messages = []
                    st.success("PDF successfully add ho gayi!")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.doc_info:
        name, n_pages, n_chunks = st.session_state.doc_info
        st.info(f"**{name}**\n\nPages: {n_pages}  \nChunks: {n_chunks}")

    if st.button("🗑️ Chat clear karein", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------
st.title("🤖 Gemini RAG App")
tab_chat, tab_image, tab_text = st.tabs(
    ["💬 Chat with PDF", "🖼️ Image Describe", "✨ Simple Prompt"]
)

# ---------- Tab 1: RAG chat ----------
with tab_chat:
    if st.session_state.vector_store is None:
        st.info("👈 Sidebar se PDF upload karke **Process PDF** dabayein.")
    else:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(msg["sources"], 1):
                            st.markdown(f"**Chunk {i}** (page {src['page']})")
                            st.caption(src["text"])

        query = st.chat_input("PDF ke baare mein sawal poochein...")
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Soch raha hoon..."):
                    try:
                        answer, results = answer_question(
                            st.session_state.vector_store, query, top_k
                        )
                    except Exception as e:
                        answer, results = f"Error: {e}", []
                st.markdown(answer)

                sources = [
                    {
                        "page": doc.metadata.get("page", "?"),
                        "text": doc.page_content,
                    }
                    for doc in results
                ]
                if sources:
                    with st.expander("📚 Sources"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Chunk {i}** (page {src['page']})")
                            st.caption(src["text"])

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

# ---------- Tab 2: Image description ----------
with tab_image:
    st.subheader("Image upload karein")
    img_file = st.file_uploader(
        "Image select karein", type=["png", "jpg", "jpeg", "webp"], key="img"
    )
    img_prompt = st.text_input("Prompt", value="Describe this image")

    if img_file:
        img = Image.open(img_file)
        st.image(img, caption=img_file.name, width=400)

        if st.button("Describe karein", type="primary"):
            with st.spinner("Image analyze ho rahi hai..."):
                try:
                    response = client.models.generate_content(
                        model=MODEL, contents=[img_prompt, img]
                    )
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")

# ---------- Tab 3: Simple prompt ----------
with tab_text:
    st.subheader("Gemini se kuch bhi poochein")
    simple_prompt = st.text_area(
        "Prompt", value="Explain Artificial Intelligence in simple words.", height=120
    )
    if st.button("Generate", type="primary", key="gen"):
        with st.spinner("Generate ho raha hai..."):
            try:
                response = client.models.generate_content(
                    model=MODEL, contents=simple_prompt
                )
                st.markdown(response.text)
            except Exception as e:
                st.error(f"Error: {e}")