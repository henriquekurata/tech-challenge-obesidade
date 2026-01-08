
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

# ===============================
# Criar abas
# ===============================
tab1, tab2, tab3, tab4 = st.tabs([
    "Problema de Negócio", 
    "Análise Exploratória", 
    "Sistema Preditivo", 
    "Conclusões"
])

# ===============================
# Estilo para os títulos dentro das abas
# ===============================
titulo_estilo = """
    <div style="
        background-color: #f0f2f6; 
        padding: 10px; 
        border-radius: 8px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    ">
        <h2 style="color: #0f4c81; font-weight: bold;">{}</h2>
    </div>
"""

# Exemplo de uso dentro de cada aba:
with tab1:
    st.markdown(titulo_estilo.format("Problema de Negócio"), unsafe_allow_html=True)
    st.markdown("""
    Aqui você explica o contexto do problema, a importância da obesidade, a base de dados usada e os objetivos do sistema preditivo.
    """)

with tab2:
    st.markdown(titulo_estilo.format("Análise Exploratória"), unsafe_allow_html=True)
    st.markdown("Gráficos e insights sobre os dados entram aqui.")

with tab3:
    st.markdown(titulo_estilo.format("Sistema Preditivo"), unsafe_allow_html=True)
    st.markdown("Formulário para previsão de obesidade aqui.")

with tab4:
    st.markdown(titulo_estilo.format("Conclusões"), unsafe_allow_html=True)
    st.markdown("Resumo do projeto, insights do modelo e recomendações.")


# ===============================
# ABA 1 — PROBLEMA DE NEGÓCIO
# ===============================
# ===============================
# ABA 1 — PROBLEMA DE NEGÓCIO
# ===============================
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
    As opções "Pouco", "Moderado" e "Muito" incluem valores de referência para facilitar a escolha.
    """)
    st.markdown("\n")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        # ---------------------
        # Coluna 1 — Dados Pessoais
        # ---------------------
        with col1:
            st.subheader("Dados Pessoais")
            genero = st.selectbox("Gênero", ["Masculino", "Feminino"])
            idade = st.slider("Idade", 10, 80, 30)
            altura = st.slider("Altura (m)", 1.40, 2.10, 1.70)
            peso = st.slider("Peso (kg)", 40.0, 160.0, 70.0)

        # ---------------------
        # Coluna 2 — Hábitos Alimentares
        # ---------------------
        with col2:
            st.subheader("Hábitos Alimentares")
            historico_familiar = st.selectbox("Histórico familiar de obesidade", ["sim", "não"])
            consumo_calorico = st.selectbox("Consumo frequente de alimentos calóricos", ["sim", "não"])

            fcvc_opcoes = ["Pouco (1 vez/dia)", "Moderado (2 vezes/dia)", "Muito (3 vezes/dia)"]
            fcvc_cat = st.selectbox("Consumo de vegetais (FCVC)", fcvc_opcoes)
            fcvc = {"Pouco (1 vez/dia)": 1, "Moderado (2 vezes/dia)": 2, "Muito (3 vezes/dia)": 3}[fcvc_cat]

            ncp_opcoes = ["Pouco (1 refeição)", "Moderado (2-3 refeições)", "Muito (4 refeições)"]
            ncp_cat = st.selectbox("Número de refeições (NCP)", ncp_opcoes)
            ncp = {"Pouco (1 refeição)": 1, "Moderado (2-3 refeições)": 2, "Muito (4 refeições)": 3}[ncp_cat]

            # Consumo de água em litros
            ch2o_opcoes = ["Pouco (0,5 L/dia)", "Moderado (1,0 L/dia)", "Muito (1,5 L/dia)"]
            ch2o_cat = st.selectbox("Consumo diário de água (CH2O)", ch2o_opcoes)
            ch2o = {"Pouco (0,5 L/dia)": 1, "Moderado (1,0 L/dia)": 2, "Muito (1,5 L/dia)": 3}[ch2o_cat]

        # ---------------------
        # Coluna 3 — Atividade e Estilo de Vida
        # ---------------------
        with col3:
            st.subheader("Atividade e Estilo de Vida")
            faf_opcoes = ["Pouco (0-1x/semana)", "Moderado (2x/semana)", "Muito (3x+/semana)"]
            faf_cat = st.selectbox("Atividade física (FAF)", faf_opcoes)
            faf = {"Pouco (0-1x/semana)": 0, "Moderado (2x/semana)": 1, "Muito (3x+/semana)": 2}[faf_cat]

            tue_opcoes = ["Pouco (0-1h/dia)", "Moderado (1-2h/dia)", "Muito (3h+/dia)"]
            tue_cat = st.selectbox("Uso de tecnologia (TUE)", tue_opcoes)
            tue = {"Pouco (0-1h/dia)": 0, "Moderado (1-2h/dia)": 1, "Muito (3h+/dia)": 2}[tue_cat]

            calc = st.selectbox("Consumo de álcool", ["não", "Às vezes", "Frequentemente", "Sempre"])
            caec = st.selectbox("Alimentação entre refeições", ["não", "Às vezes", "Frequentemente", "Sempre"])
            fuma = st.selectbox("Fuma?", ["sim", "não"])
            monitora_calorias = st.selectbox("Monitora calorias?", ["sim", "não"])
            transporte = st.selectbox(
                "Meio de transporte",
                ["Transporte Público", "Caminhada", "Automóvel", "Moto", "Bicicleta"]
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
            st.success(f"✅ O nível de obesidade previsto é: **{previsao}**")




# ===============================
# ABA 4 — CONCLUSÕES
# ===============================
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
  
