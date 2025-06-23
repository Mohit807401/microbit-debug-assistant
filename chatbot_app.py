# 📦 Import Libraries
import streamlit as st
import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# 🧠 STEP 1: Load the FAISS Vector DB
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("microbit_faiss_db", embeddings=embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# ⚡ STEP 2: Define the OpenRouter LLM call
def call_openrouter_llm(prompt):
    api_key = os.getenv("OPENROUTER_API_KEY") or st.secrets["OPENROUTER_API_KEY"]

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "mistralai/mistral-7b-instruct",  # or try gpt-3.5-turbo
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# 🪛 STEP 3: Optional fallback (local flan-t5-small for offline testing)
fallback_llm = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

# 🎛️ STEP 4: Streamlit UI
st.set_page_config(page_title="Microbit Debugging Assistant", page_icon="🛠️")
st.title("🛠️ Microbit Debugging Assistant")
st.markdown("Ask me any question related to Microbit issues or troubleshooting.")

query = st.text_input("🔍 Type your question here:")

if query:
    # 🔍 STEP 5: Get top 2 documents only, and truncate long text
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs[:2]])[:1500]

    # 🧠 Build prompt
    prompt = f"""Use the following context to answer the question clearly and helpfully.

Context:
{context}

Question:
{query}
"""

    # 🤖 STEP 6: Call OpenRouter (or fallback)
    with st.spinner("Thinking..."):
        try:
            answer = call_openrouter_llm(prompt)
        except Exception as e:
            st.warning("⚠️ OpenRouter failed. Falling back to local model.")
            answer = fallback_llm(prompt, max_length=300)[0]['generated_text']

    # ✅ Show the answer
    st.markdown("### ✅ Answer:")
    st.success(answer)
