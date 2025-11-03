import streamlit as st

app_title = "Tennis ATP Match Predictor"

st.set_page_config(page_title=app_title, page_icon=":material/sports_tennis:", layout="wide")


home = st.Page('./home.py', title='Home', icon=':material/home:')
prediction_page = st.Page('./front.py',title='Fazer previsão')
about_page = st.Page('./sobre.py',title='Sobre')
github_page = st.Page('./github.py',title='GitHub')

pg = st.navigation(pages={
    'Home':[home],
    'Fazer previsão':[prediction_page], 
    'Outros': [about_page, github_page]
    })


try:
    pg.run()
except Exception as e:
    st.error(f"An error occurred: {e}")
