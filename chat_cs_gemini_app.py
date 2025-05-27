
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai

# CONFIGURAÇÕES
genai.configure(api_key="SUA_CHAVE_GEMINI")

# MODELO PARA EMBEDDINGS
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# TÍTULO
st.set_page_config(page_title="Chat CS Inteligente", layout="wide")
st.title("🔍 Chat Inteligente sobre Clientes e Métricas de CS")

# CARREGAR DADOS
uploaded_file = st.file_uploader("📂 Envie sua planilha (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Planilha carregada com sucesso!")
    st.dataframe(df.head())

    # TRANSFORMAR EM TEXTO
    textos = df.apply(lambda row: " | ".join([f"{col}: {row[col]}" for col in df.columns]), axis=1).tolist()
    embeddings = embedder.encode(textos)

    # INDEXAÇÃO COM FAISS
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # CAIXA DE PERGUNTA
    pergunta = st.text_area("🗣️ Pergunte algo sobre os dados:", height=100)
    if st.button("Consultar IA") and pergunta:
        with st.spinner("Consultando..."):
            pergunta_emb = embedder.encode([pergunta])
            _, indices = index.search(np.array(pergunta_emb), k=5)
            trechos = [textos[i] for i in indices[0]]

            contexto = "\n".join(trechos)

            prompt = f"""
Você é um analista de Customer Success. Responda em português claro com base nos dados abaixo:

{contexto}

Pergunta: {pergunta}
"""

            try:
                resposta = genai.GenerativeModel("gemini-pro").generate_content(prompt)
                st.markdown("### 🤖 Resposta da IA:")
                st.write(resposta.text)
            except Exception as e:
                st.error("Erro ao consultar o Gemini: " + str(e))
