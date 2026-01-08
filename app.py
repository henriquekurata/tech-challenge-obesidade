
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Carregar dataset e modelo
df = pd.read_csv("Obesity.csv")
best_model = joblib.load("modelo_obesidade.pkl")

# Configuração da página
st.set_page_config(page_title="Sistema Preditivo de Obesidade", layout="wide")

# Criar abas
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Problema de Negócio", 
    "2️⃣ Análise Exploratória", 
    "3️⃣ Sistema Preditivo", 
    "4️⃣ Conclusões"
])

# ===============================
# ABA 1 — PROBLEMA DE NEGÓCIO
# ===============================
with tab1:
    st.header("Contextualização e Problema de Negócio")
    st.markdown("""
    O objetivo deste projeto é desenvolver um **sistema preditivo de obesidade**, auxiliando equipes médicas na identificação do nível de obesidade de pacientes.

    A obesidade é uma condição multifatorial, influenciada por:
    - Hábitos alimentares
    - Nível de atividade física
    - Dados demográficos e físicos
    - Histórico familiar

    O sistema busca fornecer informações consistentes, baseadas em dados, para apoiar decisões clínicas.
    """)

# ===============================
# ABA 2 — ANÁLISE EXPLORATÓRIA
# ===============================
with tab2:
    st.header("Análise Exploratória dos Dados")

    st.markdown("### Distribuição das classes de obesidade")
    st.bar_chart(df['Obesity'].value_counts())

    st.markdown("### Estatísticas gerais das variáveis numéricas")
    st.dataframe(df.describe())

    # Heatmap de correlação
    st.markdown("### Correlação entre variáveis preditivas")
    st.markdown("""
    O gráfico abaixo mostra a correlação entre as variáveis numéricas usadas no sistema preditivo.
    Valores próximos de 1 ou -1 indicam forte correlação positiva ou negativa, respectivamente.
    """)
    variaveis = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
    dados_corr = df[variaveis]
    corr_matrix = dados_corr.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlação entre variáveis do sistema preditivo")
    plt.tight_layout()
    st.pyplot(plt)

# ===============================
# ABA 3 — SISTEMA PREDITIVO
# ===============================
with tab3:
    st.header("Sistema Preditivo de Obesidade")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        # Coluna 1
        with col1:
            Gender = st.selectbox("Gênero", ["Male", "Female"])
            Age = st.slider("Idade", 10, 80, 30)
            Height = st.slider("Altura (m)", 1.40, 2.10, 1.70)
            Weight = st.slider("Peso (kg)", 40.0, 160.0, 70.0)

        # Coluna 2
        with col2:
            family_history = st.selectbox("Histórico familiar de obesidade", ["yes", "no"])
            FAVC = st.selectbox("Consumo frequente de alimentos calóricos", ["yes", "no"])

            # FCVC — Consumo de vegetais
            st.write("**Consumo de vegetais (FCVC)**")
            st.write("Pouco = 1-2 porções/dia, Moderado = 3-4, Muito = 5+")
            FCVC_cat = st.selectbox("Selecione o consumo de vegetais", ["Pouco", "Moderado", "Muito"])
            FCVC = {"Pouco":1, "Moderado":2, "Muito":3}[FCVC_cat]

            # NCP — Número de refeições principais
            st.write("**Número de refeições principais (NCP)**")
            st.write("Pouco = 1 refeição, Moderado = 2-3, Muito = 4 ou mais")
            NCP_cat = st.selectbox("Selecione o número de refeições principais", ["Pouco", "Moderado", "Muito"])
            NCP = {"Pouco":1, "Moderado":2, "Muito":3}[NCP_cat]

        # Coluna 3
        with col3:
            # CH2O — Consumo diário de água
            st.write("**Consumo diário de água (CH2O)**")
            st.write("Pouco = <1 L/dia, Moderado = 1-2 L/dia, Muito = >2 L/dia")
            CH2O_cat = st.selectbox("Selecione o consumo de água", ["Pouco", "Moderado", "Muito"])
            CH2O = {"Pouco":1, "Moderado":2, "Muito":3}[CH2O_cat]

            # FAF — Frequência de atividade física
            st.write("**Frequência de atividade física (FAF)**")
            st.write("Pouco = 0-1 vez/semana, Moderado = 2-3, Muito = 4+")
            FAF_cat = st.selectbox("Selecione a frequência de atividade física", ["Pouco", "Moderado", "Muito"])
            FAF = {"Pouco":0, "Moderado":1, "Muito":2}[FAF_cat]

            # TUE — Uso de tecnologia
            st.write("**Uso de tecnologia (TUE)**")
            st.write("Pouco = 0-1h, Moderado = 1-2h, Muito = 2+ h")
            TUE_cat = st.selectbox("Selecione o uso de tecnologia", ["Pouco", "Moderado", "Muito"])
            TUE = {"Pouco":0, "Moderado":1, "Muito":2}[TUE_cat]

            # Outras variáveis categóricas
            CALC = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
            CAEC = st.selectbox("Alimentação entre refeições", ["no", "Sometimes", "Frequently", "Always"])
            SMOKE = st.selectbox("Fuma?", ["yes", "no"])
            SCC = st.selectbox("Monitora calorias?", ["yes", "no"])
            MTRANS = st.selectbox(
                "Meio de transporte",
                ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
            )

        # Botão de previsão
        submitted = st.form_submit_button("Prever Nível de Obesidade")
        if submitted:
            input_data = pd.DataFrame([{
                "Gender": Gender,
                "Age": Age,
                "Height": Height,
                "Weight": Weight,
                "family_history": family_history,
                "FAVC": FAVC,
                "FCVC": FCVC,
                "NCP": NCP,
                "CH2O": CH2O,
                "FAF": FAF,
                "TUE": TUE,
                "CALC": CALC,
                "CAEC": CAEC,
                "SMOKE": SMOKE,
                "SCC": SCC,
                "MTRANS": MTRANS
            }])
            prediction = best_model.predict(input_data)[0]
            st.success(f"✅ O nível de obesidade previsto é: **{prediction}**")

# ===============================
# ABA 4 — CONCLUSÕES
# ===============================
with tab4:
    st.header("Conclusões e Insights do Modelo")

    st.markdown("""
    - O modelo final escolhido foi o **Gradient Boosting Classifier**, por apresentar melhor desempenho na classificação multiclasse.
    - A acurácia do modelo atingiu mais de **75%**, cumprindo os critérios do desafio.
    - As variáveis mais relevantes foram:
        - Peso  
        - Atividade física (FAF)  
        - Hábitos alimentares (FAVC, FCVC, CAEC, CALC)  
        - Histórico familiar (family_history)
    - O sistema atua como **ferramenta de apoio à decisão clínica**, auxiliando ações preventivas, acompanhamento de pacientes e melhor gestão da saúde.
    """)
