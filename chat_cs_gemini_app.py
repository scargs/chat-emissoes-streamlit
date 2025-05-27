import streamlit as st
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import google.generativeai as genai

# Configurações da página
st.set_page_config(page_title="Chat Inteligente sobre Clientes e Métricas de CS", layout="wide")
st.title("🔍 Chat Inteligente sobre Clientes e Métricas de CS")

# Conectar ao Google Sheets via Service Account (armazenado em st.secrets)
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
client = gspread.authorize(creds)

# Ler dados da planilha
sheet = client.open("clientes_mrr_emissoes").sheet1
data = sheet.get_all_records()
df = pd.DataFrame(data)

# Exibir preview dos dados
with st.expander("👁️ Visualizar primeiros registros da base"):
    st.dataframe(df.head())

# Campo de pergunta
st.subheader("📨 Faça uma pergunta sobre os dados:")
pergunta = st.text_area("Exemplo: Quais clientes mais emitiram documentos no mês passado?", height=100)

# Configurar Gemini
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-pro")

# Botão para consultar
if st.button("💬 Consultar IA") and pergunta:
    with st.spinner("Processando..."):
        contexto = df.to_csv(index=False)
        prompt = f"""
Você é um analista de Customer Success. Abaixo estão os dados dos clientes da empresa no formato CSV:

{contexto[:20000]}  # Limitando para não estourar o contexto

Com base nesses dados, responda à pergunta abaixo de forma clara, objetiva e em português:

{pergunta}
"""
        resposta = model.generate_content(prompt)
        st.markdown("### 🤖 Resposta da IA:")
        st.write(resposta.text)

# Gráfico opcional
with st.expander("📊 Gráfico de exemplo (Top 10 por emissão)"):
    if "cliente" in df.columns and "emissões" in df.columns:
        df_top = df.sort_values(by="emissões", ascending=False).head(10)
        st.bar_chart(df_top.set_index("cliente")["emissões"])
