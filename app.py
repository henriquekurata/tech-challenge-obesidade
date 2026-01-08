
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
# ABA 2 — ANÁLISE EXPLORATÓRIA (duas colunas)
# ===============================
with tab2:
    st.header("Análise Exploratória dos Dados")
    st.markdown("\n")

    # Criar duas colunas
    col1, col2 = st.columns(2)

    # ---------------------
    # Coluna 1
    # ---------------------
    with col1:
        # Distribuição das classes de obesidade
        st.markdown("### Distribuição das classes de obesidade")
        plt.figure(figsize=(4, 3))
        sns.countplot(data=df, x="Obesity", palette="pastel", order=df['Obesity'].value_counts().index)
        plt.title("Pacientes por obesidade", fontsize=10)
        plt.xlabel("Obesidade", fontsize=9)
        plt.ylabel("Contagem", fontsize=9)
        plt.xticks(rotation=30, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(plt)

        st.markdown("\n")

        # Scatter plot: Idade x Peso
        st.markdown("### Idade x Peso")
        plt.figure(figsize=(4, 3))
        sns.scatterplot(data=df, x="Age", y="Weight", hue="Obesity", palette="bright", s=40)
        plt.title("Idade vs Peso", fontsize=10)
        plt.xlabel("Idade", fontsize=9)
        plt.ylabel("Peso (kg)", fontsize=9)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(fontsize=7, loc='upper right')
        plt.tight_layout()
        st.pyplot(plt)

    # ---------------------
    # Coluna 2
    # ---------------------
    with col2:
        # Obesidade x Frequência de Atividade Física (FAF)
        st.markdown("### Obesidade x Atividade Física")
        plt.figure(figsize=(4, 3))
        sns.boxplot(data=df, x="Obesity", y="FAF", palette="Set2")
        plt.title("FAF por obesidade", fontsize=10)
        plt.xlabel("Obesidade", fontsize=9)
        plt.ylabel("FAF", fontsize=9)
        plt.xticks(rotation=30, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(plt)

        st.markdown("\n")

        # Heatmap de correlação
        st.markdown("### Correlação entre variáveis")
        variaveis = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
        dados_corr = df[variaveis]
        corr_matrix = dados_corr.corr()
        plt.figure(figsize=(5, 4))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True)
        plt.title("Correlação das variáveis", fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(plt)


# ===============================
# ABA 3 — SISTEMA PREDITIVO
# ===============================
with tab3:
    st.header("Sistema Preditivo de Obesidade")
    st.markdown("""
    Preencha os dados do paciente abaixo e clique em **Prever Nível de Obesidade**.
    As opções "Pouco", "Moderado" e "Muito" incluem valores de referência.
    """)
    st.markdown("\n")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        # ---------------------
        # Coluna 1 — Dados Pessoais
        # ---------------------
        with col1:
            Gender = st.selectbox("Gênero", ["Male", "Female"])
            Age = st.slider("Idade", 10, 80, 30)
            Height = st.slider("Altura (m)", 1.40, 2.10, 1.70)
            Weight = st.slider("Peso (kg)", 40.0, 160.0, 70.0)

        # ---------------------
        # Coluna 2 — Hábitos Alimentares
        # ---------------------
        with col2:
            family_history = st.selectbox("Histórico familiar de obesidade", ["yes", "no"])
            FAVC = st.selectbox("Consumo frequente de alimentos calóricos", ["yes", "no"])

            FCVC_options = ["Pouco (1 vez/dia)", "Moderado (2 vezes/dia)", "Muito (3 vezes/dia)"]
            FCVC_cat = st.selectbox("Consumo de vegetais (FCVC)", FCVC_options)
            FCVC = {"Pouco (1 vez/dia)": 1, "Moderado (2 vezes/dia)": 2, "Muito (3 vezes/dia)": 3}[FCVC_cat]

            NCP_options = ["Pouco (1 refeição)", "Moderado (2-3 refeições)", "Muito (4 refeições)"]
            NCP_cat = st.selectbox("Número de refeições (NCP)", NCP_options)
            NCP = {"Pouco (1 refeição)": 1, "Moderado (2-3 refeições)": 2, "Muito (4 refeições)": 3}[NCP_cat]

            CH2O_options = ["Pouco (1 copo/dia)", "Moderado (2 copos/dia)", "Muito (3 copos/dia)"]
            CH2O_cat = st.selectbox("Consumo diário de água (CH2O)", CH2O_options)
            CH2O = {"Pouco (1 copo/dia)": 1, "Moderado (2 copos/dia)": 2, "Muito (3 copos/dia)": 3}[CH2O_cat]

        # ---------------------
        # Coluna 3 — Atividade e Estilo de Vida
        # ---------------------
        with col3:
            FAF_options = ["Pouco (0-1x/semana)", "Moderado (2x/semana)", "Muito (3x+/semana)"]
            FAF_cat = st.selectbox("Atividade física (FAF)", FAF_options)
            FAF = {"Pouco (0-1x/semana)": 0, "Moderado (2x/semana)": 1, "Muito (3x+/semana)": 2}[FAF_cat]

            TUE_options = ["Pouco (0-1h/dia)", "Moderado (1-2h/dia)", "Muito (3h+/dia)"]
            TUE_cat = st.selectbox("Uso de tecnologia (TUE)", TUE_options)
            TUE = {"Pouco (0-1h/dia)": 0, "Moderado (1-2h/dia)": 1, "Muito (3h+/dia)": 2}[TUE_cat]

            CALC = st.selectbox("Consumo de álcool", ["no", "Sometimes", "Frequently", "Always"])
            CAEC = st.selectbox("Alimentação entre refeições", ["no", "Sometimes", "Frequently", "Always"])
            SMOKE = st.selectbox("Fuma?", ["yes", "no"])
            SCC = st.selectbox("Monitora calorias?", ["yes", "no"])
            MTRANS = st.selectbox(
                "Meio de transporte",
                ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
            )

        st.markdown("\n")
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

            # Previsão usando o modelo carregado
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
