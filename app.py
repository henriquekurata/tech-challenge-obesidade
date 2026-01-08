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
    "Problema de Negócio", 
    "Análise Exploratória", 
    "Sistema Preditivo", 
    "Conclusões"
])



# ABA 1 — PROBLEMA DE NEGÓCIO
with tab1:
    st.header("Contexto e Problema de Negócio")
    
    st.markdown("""
    A obesidade é uma condição multifatorial que representa um importante problema de saúde pública. 
    Ela está associada a diversas doenças crônicas, como diabetes, hipertensão, doenças cardiovasculares e outras complicações de saúde. 
    Identificar precocemente o nível de obesidade de um indivíduo pode auxiliar profissionais de saúde a recomendar intervenções adequadas, 
    prevenindo o agravamento de condições médicas.

    Este projeto faz parte do **Tech Challenge – Fase 4 da pós-graduação em Data Analytics** e tem como objetivo desenvolver um **sistema preditivo de obesidade**. 
    A ferramenta foi criada para **auxiliar a equipe médica** na identificação do nível de obesidade de pacientes, 
    usando dados demográficos, físicos e comportamentais de forma objetiva e baseada em dados.

    A base utilizada é o conjunto de dados **Obesity.csv**, que contém informações sobre idade, gênero, peso, altura, hábitos alimentares, nível de atividade física, histórico familiar e outros fatores que influenciam o desenvolvimento da obesidade.

    O problema é tratado como uma **classificação multiclasse**, pois o nível de obesidade é categorizado em diferentes classes:  
    por exemplo, "Peso Normal", "Sobrepeso" e "Obesidade" (em diferentes níveis).  

    O objetivo do sistema é fornecer **previsões precisas e rápidas**, apoiando decisões clínicas e contribuindo para ações preventivas e planejamento de tratamentos.
    """)



# ABA 2 — ANÁLISE EXPLORATÓRIA
with tab2:
    st.header("Análise Exploratória dos Dados")
    st.markdown("\n")

    # Criar duas colunas
    col1, col2 = st.columns(2)

    #Coluna 1 
    with col1:
        # Distribuição das classes de obesidade
        st.markdown("### Distribuição das classes de obesidade")
        plt.figure(figsize=(4, 3))
        sns.countplot(
            data=df,
            x="Obesity",
            palette="pastel",
            order=df['Obesity'].value_counts().index
        )
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
        sns.scatterplot(
            data=df,
            x="Age",
            y="Weight",
            hue="Obesity",
            palette="bright",
            s=40
        )
        plt.title("Idade vs Peso", fontsize=10)
        plt.xlabel("Idade (anos)", fontsize=9)
        plt.ylabel("Peso (kg)", fontsize=9)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend(fontsize=7, loc='upper right')
        plt.tight_layout()
        st.pyplot(plt)

    #Coluna 2
    with col2:
        # Obesidade x Frequência de Atividade Física (FAF)
        st.markdown("### Obesidade x Atividade Física")
        plt.figure(figsize=(4, 3))
        sns.boxplot(
            data=df,
            x="Obesity",
            y="FAF",
            palette="Set2"
        )
        plt.title("FAF por obesidade", fontsize=10)
        plt.xlabel("Obesidade", fontsize=9)
        plt.ylabel("Frequência de Atividade Física", fontsize=9)
        plt.xticks(rotation=30, fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(plt)

        st.markdown("\n")

        # Heatmap de correlação
        st.markdown("### Correlação entre variáveis numéricas")
        variaveis = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
        dados_corr = df[variaveis]
        corr_matrix = dados_corr.corr()
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            linewidths=0.5,
            cbar=True
        )
        plt.title("Correlação das variáveis", fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        st.pyplot(plt)




# ABA 3 — SISTEMA PREDITIVO
with tab3:
    st.markdown("""
    <div style="
        background-color: #f0f2f6; 
        padding: 10px; 
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    ">
        <h2 style="color: #0f4c81; font-weight: bold;">Sistema Preditivo de Obesidade</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    Preencha os dados do paciente abaixo e clique em **Prever Nível de Obesidade**.
    As opções apresentam **valores de referência** para facilitar a escolha.
    """)

    st.markdown("\n")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        #Coluna 1 — Dados Pessoais 
        with col1:
            st.subheader("Dados Pessoais")
            genero = st.selectbox("Gênero", ["Masculino", "Feminino"])
            idade = st.slider("Idade (anos)", 14, 61, 30)
            altura = st.slider("Altura (m)", 1.45, 1.98, 1.70)
            peso = st.slider("Peso (kg)", 39, 173, 70)

        #Coluna 2 — Hábitos Alimentares 
        with col2:
            st.subheader("Hábitos Alimentares")
            historico_familiar = st.selectbox("Histórico familiar de obesidade", ["sim", "não"])
            consumo_calorico = st.selectbox("Consumo frequente de alimentos calóricos", ["sim", "não"])

            fcvc_opcoes = ["Raramente (1 vez/dia)", "Às vezes (2 vezes/dia)", "Sempre (3 vezes/dia)"]
            fcvc_cat = st.selectbox("Consumo de vegetais (FCVC)", fcvc_opcoes)
            fcvc = {"Raramente (1 vez/dia)": 1, "Às vezes (2 vezes/dia)": 2, "Sempre (3 vezes/dia)": 3}[fcvc_cat]

            ncp_opcoes = ["Uma (1 refeição)", "Duas (2 refeições)", "Três (3 refeições)", "Quatro ou mais (4+)"]
            ncp_cat = st.selectbox("Número de refeições (NCP)", ncp_opcoes)
            ncp = {"Uma (1 refeição)": 1, "Duas (2 refeições)": 2, "Três (3 refeições)": 3, "Quatro ou mais (4+)": 4}[ncp_cat]

            ch2o_opcoes = ["<1 L/dia", "1–2 L/dia", ">2 L/dia"]
            ch2o_cat = st.selectbox("Consumo diário de água (CH2O)", ch2o_opcoes)
            ch2o = {"<1 L/dia": 1, "1–2 L/dia": 2, ">2 L/dia": 3}[ch2o_cat]

        #Coluna 3 — Atividade e Estilo de Vida 
        with col3:
            st.subheader("Atividade e Estilo de Vida")
            faf_opcoes = ["Nenhuma (0)", "1–2×/sem", "3–4×/sem", "5×/sem ou mais"]
            faf_cat = st.selectbox("Atividade física (FAF)", faf_opcoes)
            faf = {"Nenhuma (0)": 0, "1–2×/sem": 1, "3–4×/sem": 2, "5×/sem ou mais": 3}[faf_cat]

            tue_opcoes = ["0–2 h/dia", "3–5 h/dia", ">5 h/dia"]
            tue_cat = st.selectbox("Uso de tecnologia (TUE)", tue_opcoes)
            tue = {"0–2 h/dia": 0, "3–5 h/dia": 1, ">5 h/dia": 2}[tue_cat]

            calc = st.selectbox("Consumo de álcool", ["não", "Às vezes", "Frequentemente", "Sempre"])
            caec = st.selectbox("Alimentação entre refeições", ["não", "Às vezes", "Frequentemente", "Sempre"])
            fuma = st.selectbox("Fuma?", ["sim", "não"])
            monitora_calorias = st.selectbox("Monitora calorias?", ["sim", "não"])
            transporte = st.selectbox(
                "Meio de transporte",
                ["Automóvel", "Moto", "Bicicleta", "Transporte Público", "Caminhada"]
            )

        st.markdown("\n")
        botao = st.form_submit_button("Prever Nível de Obesidade")

        if botao:
            input_data = pd.DataFrame([{
                "Gender": genero,
                "Age": idade,
                "Height": altura,
                "Weight": peso,
                "family_history": historico_familiar,
                "FAVC": consumo_calorico,
                "FCVC": fcvc,
                "NCP": ncp,
                "CH2O": ch2o,
                "FAF": faf,
                "TUE": tue,
                "CALC": calc,
                "CAEC": caec,
                "SMOKE": fuma,
                "SCC": monitora_calorias,
                "MTRANS": transporte
            }])

            previsao = best_model.predict(input_data)[0]
            st.success(f"O nível de obesidade previsto é: **{previsao}**")





# ABA 4 — CONCLUSÕES
with tab4:
    st.header("Conclusões e Insights do Modelo")

    st.markdown("""
    - O modelo final escolhido foi o **Gradient Boosting Classifier**, por apresentar melhor desempenho na classificação multiclasse.
    - A acurácia do modelo atingiu mais de **75%**.
    - As variáveis mais relevantes foram:
        - Peso  
        - Atividade física (FAF)  
        - Hábitos alimentares (FAVC, FCVC, CAEC, CALC)  
        - Histórico familiar (family_history)
        """)
