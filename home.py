import streamlit as st

st.set_page_config(page_title="Home - Previsor de Tênis")

st.title("Previsor de partidas de Tênis")
st.markdown(
    "Bem-vindo ao previsor de partidas de tênis ATP! "
    "Este projeto usa dados históricos da ATP para treinar modelos que estimam a probabilidade de vitória "
    "entre jogadores. Use as páginas abaixo para fazer previsões, explorar os dados e ver detalhes sobre o projeto."
)

st.markdown("Navegue pelas funcionalidades:")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Fazer previsão"):
        try:
            # tenta navegar trocando o parâmetro de query (compatível com apps multi-page)
            st.experimental_set_query_params(page="front.py")
            st.experimental_rerun()
        except Exception:
            # fallback: link direto (pode variar conforme sua estrutura de páginas)
            st.markdown("[Ir para Fazer previsão](front.py)")

with col2:
    if st.button("Explorar dados"):
        try:
            st.experimental_set_query_params(page="data.py")
            st.experimental_rerun()
        except Exception:
            st.markdown("[Ir para Explorar dados](data.py)")

with col3:
    if st.button("Sobre o projeto"):
        try:
            st.experimental_set_query_params(page="about.py")
            st.experimental_rerun()
        except Exception:
            st.markdown("[Ir para Sobre](about.py)")

st.write("---")
st.write("Dica: se sua versão do Streamlit oferecer st.page_link, você também pode usá-lo para criar links de página.")