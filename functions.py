import streamlit as st
import pandas as pd
import requests

from model_presentation import make_feature_row


@st.cache_data(ttl='1d')
def get_players():
    players = pd.read_csv('./data/atp_players.csv', low_memory=False)
    players['name_full'] = players['name_first'] + ' ' + players['name_last']
    return players

@st.cache_data(ttl='1d')
def get_player_photo(player_name: str) -> str:
    res = requests.get(f'https://www.tennisabstract.com/cgi-bin/player.cgi?p={player_name.replace(" ", "")}')
    img_line_start = res.text.find('var photog')
    img_start = res.text.find(" = \'", img_line_start)
    img_end = res.text.find("\';", img_start)
    img_line = res.text[img_start+4:img_end]
    if img_line == '' or len(img_line)>20:
        return None
    player_name = player_name.lower()
    url = f'https://www.tennisabstract.com/photos/{player_name.replace(" ", "_")}-{img_line}.jpg'
    return url

def get_player_id(name:str, df):
    name = name.title()
    try:
        return df[df['name_full']==name]['player_id'].values[0]
    except Exception as e:
        raise ValueError(f"Player '{name}' not found in dataframe") from e

def previsao_real(p1:int,p2:int, surface, best_of, draw_size, model, state):
    surface_translation = {'Rápida':'Hard', 'Saibro':'Clay', 'Grama':'Grass'}
    surface = surface_translation[surface]
    context = {
    "tourney_date": pd.Timestamp("2025-06-01"),
    "surface": surface,
    "best_of": best_of,
    "draw_size": draw_size
    }
    features_1row = make_feature_row(player1_id=p1, player2_id=p2, context=context, state=state)
    proba = model.predict_proba(features_1row.drop(['player1_id', 'player2_id'], axis=1))[:,1][0]
    # proba = proba*0.95 if surface == 'Clay' else proba*1.05 if surface == 'Grass' else proba
    return proba, 1-proba