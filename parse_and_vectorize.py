from docx import Document
import json
import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

print("📂 Current directory:", os.getcwd())
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Parse DOCX
def load_docx(path):
    print(f"📄 Reading file: {path}")
    doc = Document(path)
    paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
    print(f"📄 Total paragraphs read: {len(paragraphs)}")
    return paragraphs


# Step 2: Extract cases
def extract_cases(text_blocks):
    cases = []
    current_case = {}
    capture = None

    for line in text_blocks:
        line_lower = line.lower()
        case_match = re.match(r"(case\s+\d+|issue\s+[A-Z]):?\s*(.*)", line, re.IGNORECASE)

        if case_match:
            if current_case:
                cases.append(current_case)
                current_case = {}
            current_case["title"] = line
            capture = None
            continue

        if "symptom" in line_lower:
            capture = "symptoms"
            current_case[capture] = []
        elif "cause" in line_lower:
            capture = "causes"
            current_case[capture] = []
        elif "solution" in line_lower:
            capture = "solutions"
            current_case[capture] = []
        elif capture:
            current_case[capture].append(line)

    if current_case:
        cases.append(current_case)

    print(f"✅ Extracted {len(cases)} valid cases.")
    return cases


# Step 3: Convert to chunks and save to FAISS
from langchain_community.vectorstores import FAISS

def build_vector_db(cases):
    docs = []
    for i, case in enumerate(cases):
        if "title" not in case:
            print(f"⚠️ Skipping case #{i} due to missing title")
            continue

        combined = (
            f"{case['title']}\n"
            f"Symptoms: {'; '.join(case.get('symptoms', []))}\n"
            f"Causes: {'; '.join(case.get('causes', []))}\n"
            f"Solutions: {'; '.join(case.get('solutions', []))}"
        )
        docs.append(combined)

    if not docs:
        raise ValueError("No valid cases to build vector DB.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_text("\n".join(docs))

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts, embedding)
    vectorstore.save_local("microbit_faiss_db")
    print("✅ Vector DB created and saved!")




if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "debugging_microbit.docx")
    print("🔍 Looking for file in:", file_path)

    raw_text = load_docx(file_path)
    print("✅ Loaded raw text")

    structured_cases = extract_cases(raw_text)
    print("✅ Structured cases ready")

    build_vector_db(structured_cases)


