import streamlit as st
import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# -------------------------------
# 🧠 Load FAISS Vector DB
# -------------------------------
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("microbit_faiss_db", embeddings=embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# -------------------------------
# ⚡ OpenRouter Call Function
# -------------------------------
def call_openrouter_llm(prompt):
    api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets["OPENROUTER_API_KEY"]
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistralai/mistral-7b-instruct",  # Or gpt-3.5, claude-3, llama3, etc.
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
    )

    return response.json()["choices"][0]["message"]["content"]

# -------------------------------
# 🎛️ Streamlit App UI
# -------------------------------
st.set_page_config(page_title="Microbit Debugging Assistant", page_icon="🛠️")
st.title("🛠️ Microbit Debugging Assistant")
st.markdown("Ask me any question related to Microbit issues or troubleshooting.")

query = st.text_input("🔍 Type your question here:")

if query:
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""Use the following context to answer the question clearly and helpfully.

Context:
{context}

Question:
{query}
"""

    with st.spinner("Thinking..."):
        answer = call_openrouter_llm(prompt)

    st.markdown("### ✅ Answer:")
    st.success(answer)
