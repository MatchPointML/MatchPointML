from io import BytesIO
import random
import time
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageOps, ImageFont
from functions import get_player_photo, get_players
from pathlib import Path
import pandas as pd

from model_presentation import build_feature_state, make_feature_row

#=======================PREVISOES=======================

import joblib
model = joblib.load("models/tennis_winner_model.joblib")

matches_list = []
for year in range(2020, 2025):
    m = pd.read_csv(f'./data/atp_matches_{year}.csv',low_memory=False)
    matches_list.append(m)
all_matches = pd.concat(matches_list, ignore_index=True)
all_matches['tourney_date'] = pd.to_datetime(all_matches['tourney_date'], format='%Y%m%d')
state = build_feature_state(all_matches)
#======================================================



st.set_page_config(layout="wide", page_icon=':material/sports_tennis:', page_title="Previsão de Partidas de Tênis ATP")


ca,cb,cc = st.columns([1,6,1])

cb.title("Previsão de Partidas de Tênis")

players = get_players()
names = players['name_full'].tolist()

caa,cbb = cb.columns(2)
c1,c2,c3 = cb.columns(3)


player1_name = caa.selectbox("Selecione o jogador 1", options=names)
player2_name = cbb.selectbox("Selecione o jogador 2", options=names)

surface = c1.pills("Selecione o tipo de quadra", options=['Rápida', 'Saibro', 'Grama'], default='Rápida')

best_of = c2.pills("Selecione o tipo de jogo", options=['Melhor de 3', 'Melhor de 5'], default='Melhor de 3')

draw_size = c3.pills("Selecione o tamanho do quadro", options=['32', '64', '128'], default='32')


def circular_avatar(img: Image.Image, size=220) -> Image.Image:
    img = img.convert("RGBA").resize((size, size))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    # borda branca opcional
    avatar = ImageOps.fit(img, (size, size), centering=(0.5, 0.5))
    avatar.putalpha(mask)
    border = Image.new("RGBA", (size+8, size+8), (255,255,255,0))
    ImageDraw.Draw(border).ellipse((0,0,size+8,size+8), fill=(255,255,255,230))
    border.paste(avatar, (4,4), avatar)
    return border

def paste_center(bg: Image.Image, fg: Image.Image, cx: int, cy: int):
    x = int(cx - fg.width/2)
    y = int(cy - fg.height/2)
    bg.paste(fg, (x, y), fg)

def load_image_source(source):
    """Aceita caminho local, arquivo enviado ou URL"""
    if isinstance(source, str):
        if source.startswith("http"):
            resp = requests.get(source)
            try:
                return Image.open(BytesIO(resp.content))
            except:
                print(resp.content)
                return None
        else:
            return Image.open(source)
    elif hasattr(source, "read"):  # arquivo carregado no streamlit
        return Image.open(source)
    else:
        raise ValueError("Fonte de imagem inválida")

def mock_previsao():
    time.sleep(2)
    p1 = random.random()
    return p1, 1-p1

def previsao_real(p1,p2, surface, best_of, draw_size):
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
    proba = proba*0.95 if surface == 'Clay' else proba*1.05 if surface == 'Grass' else proba
    return proba, 1-proba

if 'p1' not in st.session_state:
    st.session_state['p1'] = None
    st.session_state['p2'] = None

def on_predict(p1,p2, surface, best_of, draw_size):
    p1,p2 = previsao_real(players[players['name_full']==player1_name]['player_id'].values[0],players[players['name_full']==player2_name]['player_id'].values[0], surface, best_of[-1], draw_size)
    # st.session_state['p1'] = p1
    # st.session_state['p2'] = p2
    return p1,p2

base = Image.open("images/saibro.png").convert("RGBA") if surface == 'Saibro' else Image.open("images/grama.png").convert("RGBA") if surface == 'Grama' else Image.open("images/rapida.png").convert("RGBA")

canvas = base.copy()

def placeholder(size=220):
    img = Image.open('images\placeholder.webp').convert("RGBA")
    return circular_avatar(img, size=size)

p1_img_url = get_player_photo(player1_name)
p2_img_url = get_player_photo(player2_name)
p1_img = circular_avatar(load_image_source(p1_img_url)) if p1_img_url else placeholder()
p2_img = circular_avatar(load_image_source(p2_img_url)) if p2_img_url else placeholder()

W, H = canvas.size

left_cx  = int(W * 0.22)
right_cx = int(W * 0.78)
cy       = int(H * 0.50)

paste_center(canvas, p1_img, left_cx, cy)
paste_center(canvas, p2_img, right_cx, cy)

if Path("assets/Inter-SemiBold.ttf").exists():
    FONT = ImageFont.truetype("assets/Inter-SemiBold.ttf", 64)
elif Path("/System/Library/Fonts/Supplemental/Arial.ttf").exists():
    FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 64)
else:
    FONT = ImageFont.load_default()

def draw_centered_text(im: Image.Image, text: str, cx: int, y: int):
    draw = ImageDraw.Draw(im)
    font_size = max(32, int(im.height * 0.01))
    try:
        FONT = ImageFont.truetype("arial.ttf", font_size)
    except:
        FONT = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=FONT)
    w = bbox[2] - bbox[0]
    draw.text(
        (int(cx - w/2), int(y)),
        text,
        font=FONT,
        fill=(255, 255, 255, 255),
        stroke_width=4,
        stroke_fill=(0, 0, 0, 200)
    )

if cb.button("Prever Resultado", use_container_width=True):
    with st.spinner("Calculando previsão..."):

        p1,p2 = on_predict(player1_name,player2_name, surface, best_of, draw_size)
        st.session_state['p1'] = p1
        st.session_state['p2'] = p2

paste_center(canvas, p1_img, left_cx,  cy)
paste_center(canvas, p2_img, right_cx, cy)

if st.session_state['p1'] is not None and st.session_state['p2'] is not None:
    y_text = cy + p1_img.height // 2 + 20
    draw_centered_text(canvas, f"{player1_name}: {st.session_state['p1']*100:.1f}%", left_cx,  y_text)
    draw_centered_text(canvas, f"{player2_name}: {st.session_state['p2']*100:.1f}%", right_cx, y_text)

cb.image(canvas, use_container_width =True)
