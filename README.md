# microbit-debug-assistant

# Microbit Debug Assistant 🤖

An AI-powered assistant that helps Project Guides (PGs) troubleshoot Microbit and related kits.

## Features

- Upload your Microbit debugging `.docx` sheet
- Ask natural language questions (e.g., "yellow light not glowing")
- Instant answers based on indexed cases
- Fully offline and deployable online via Streamlit Cloud

## How to Run Locally

```bash
pip install -r requirements.txt
python parse_and_vectorize.py
streamlit run chatbot_app.py
