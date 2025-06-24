import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# Load vector DB
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("microbit_faiss_db", embeddings=embedding, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Load lightweight local LLM
llm = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

def microbit_assistant(message, history):
    docs = retriever.invoke(message)  # Updated to avoid deprecation warning
    context = "\n".join([doc.page_content for doc in docs[:2]])[:1000]
    prompt = (
    "You are a helpful assistant for debugging Microbit issues.\n"
    "Based on the following troubleshooting guide, answer the user's question clearly and helpfully.\n\n"
    f"Troubleshooting Info:\n{context}\n\n"
    f"User Question: {message}\n"
    "Assistant Answer:"
)

    result = llm(prompt, max_new_tokens=256)[0]['generated_text']
    return result  # ✅ This is what ChatInterface expects


# Gradio UI
chatbot = gr.Chatbot()
demo = gr.ChatInterface(fn=microbit_assistant, chatbot=chatbot, title="🛠️ Microbit Debugging Assistant")

if __name__ == "__main__":
    demo.launch()
