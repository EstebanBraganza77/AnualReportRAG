import streamlit as st
from typing import Dict
from dotenv import load_dotenv
from agent.rag_agent import handle_question  # Asegúrate de importar correctamente tus funciones
import os
import json

# Cargar variables de entorno
load_dotenv()

# === Interfaz de Streamlit ===
st.title("📊 Annual Report Chatbot")

# Inputs
ticker = st.text_input("Company Ticker", value="DMART")
year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)
question = st.text_area("Ask a question about the annual report", height=100)

with open("section_descriptions.json", "r") as f:
    section_descriptions = json.load(f)
    section_descriptions = section_descriptions[ticker][str(year)]

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing your question..."):
            try:
                answer = handle_question(
                    question=question,
                    ticker=ticker,
                    year=year,
                    section_descriptions=section_descriptions
                )
                st.markdown("### 🧠 Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
