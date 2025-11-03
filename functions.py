import streamlit as st
import pandas as pd
import requests


@st.cache_data(ttl='1d')
def get_players():
    players = pd.read_csv('./data/atp_players.csv')
    players['name_full'] = players['name_first'] + ' ' + players['name_last']
    return players

@st.cache_data(ttl='1d')
def get_player_photo(player_name: str) -> str:
    res = requests.get(f'https://www.tennisabstract.com/cgi-bin/player.cgi?p={player_name.replace(" ", "")}')
    img_line_start = res.text.find('var photog')
    img_start = res.text.find(" = \'", img_line_start)
    img_end = res.text.find("\';", img_start)
    img_line = res.text[img_start+4:img_end]
    if img_line == '':
        return None
    player_name = player_name.lower()
    url = f'https://www.tennisabstract.com/photos/{player_name.replace(" ", "_")}-{img_line}.jpg'
    return url