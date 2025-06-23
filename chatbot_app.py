import streamlit as st
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load the vector database
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("microbit_faiss_db", embeddings=embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Load the HuggingFace model
llm = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)


# Streamlit UI
st.set_page_config(page_title="Microbit Debugging Assistant", page_icon="🛠️")
st.title("🛠️ Microbit Debugging Assistant")
st.markdown("Ask me any question related to Microbit issues ")

query = st.text_input("🔍 Type your question here:")

if query:
    # Retrieve relevant documents
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])

    # Generate response
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = llm(prompt, max_length=300)[0]['generated_text']

    # Display the answer
    st.markdown("### ✅ Answer:")
    st.write(response)
