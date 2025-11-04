import pandas as pd
import numpy as np
from collections import defaultdict, deque

df = pd.read_csv('./data/atp_matches_features_balanced.csv')
df_original = pd.read_csv('./data/atp_matches_2024.csv')
players = pd.read_csv('./data/atp_players.csv')
# ==============================
# Utilitários
# ==============================

K_BASE = 24.0

def event_importance(level, rnd):
    if level == 'G': return 1.50
    if level == 'M': return 1.25
    if level == 'F': return 1.60 if rnd in {'SF','F'} else 1.25
    if level == 'A': return 1.10
    if level == 'C': return 1.00
    return 1.00

def sets_factor(best_of):
    try:
        return 1.35 if int(best_of) >= 5 else 1.00
    except Exception:
        return 1.00

def expected_score(Ra, Rb):
    return 1.0 / (1.0 + 10 ** ((Rb - Ra) / 400.0))

def rolling_slope(seq, min_points=2):
    n = len(seq)
    if n < min_points:
        return np.nan
    x = np.arange(n, dtype=float)
    y = np.asarray(seq, dtype=float)
    xm, ym = x.mean(), y.mean()
    denom = np.sum((x - xm) ** 2)
    if denom == 0.0:
        return 0.0
    return np.sum((x - xm) * (y - ym)) / denom

def safe_mean(seq):
    return np.mean(seq) if len(seq) > 0 else np.nan

# ==============================
# 1) Construção do "estado" a partir do histórico
# ==============================

def build_feature_state(all_matches):
    """
    all_matches: DataFrame bruto da ATP (1968-2024...), com colunas:
      - tourney_date (datetime), match_num, surface, round, tourney_level, best_of, draw_size
      - winner_id, loser_id, w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced
      - l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced
      - winner_rank, winner_rank_points, loser_rank, loser_rank_points
      - winner_age, loser_age, winner_ht, loser_ht
    Retorna um dicionário "state" com tudo que precisamos para pré-jogo.
    """
    df = all_matches.copy()
    # normaliza nomes player1/player2 para varrer em ordem (player1 = winner nessa base)
    df.columns = [c.replace('w_', 'player1_').replace('winner_', 'player1_')
                    .replace('l_', 'player2_').replace('loser_', 'player2_')
                  for c in df.columns]
    # ordenação crono-estável
    df = df.sort_values(["tourney_date", "match_num"], kind="mergesort").reset_index(drop=True)

    # Estados
    elo = defaultdict(lambda: 1500.0)
    elo_hist = defaultdict(list)  # para slope
    last_rank = {}
    last_pts = {}
    last_age = {}
    last_ht = {}

    h2h = defaultdict(int)  # (a,b) -> vitórias de a sobre b
    h2h_surface = defaultdict(lambda: defaultdict(int))  # surface -> (a,b) -> vitórias

    # deques de estatísticas de saque por janela k
    Ks = [3,5,10,20,50,100]
    k_hist = defaultdict(lambda: defaultdict(lambda: {  # player -> metric -> dict por K
        3: deque(maxlen=3), 5: deque(maxlen=5), 10: deque(maxlen=10),
        20: deque(maxlen=20), 50: deque(maxlen=50), 100: deque(maxlen=100)
    }))
    # métricas
    METRICS = ("p_ace","p_df","p_1stIn","p_1stWon","p_2ndWon","p_bpSaved")

    for r in df.itertuples(index=False):
        a = r.player1_id  # winner
        b = r.player2_id  # loser

        # registra infos "last known" ANTES de atualizar
        last_rank[a] = getattr(r, "player1_rank", np.nan)
        last_rank[b] = getattr(r, "player2_rank", np.nan)
        last_pts[a]  = getattr(r, "player1_rank_points", np.nan)
        last_pts[b]  = getattr(r, "player2_rank_points", np.nan)
        last_age[a]  = getattr(r, "player1_age", np.nan)
        last_age[b]  = getattr(r, "player2_age", np.nan)
        last_ht[a]   = getattr(r, "player1_ht", np.nan)
        last_ht[b]   = getattr(r, "player2_ht", np.nan)

        # atualiza H2H (a venceu b)
        h2h[(a,b)] += 1
        surf = getattr(r, "surface", None)
        if pd.notna(surf):
            h2h_surface[surf][(a,b)] += 1

        # guarda Elo pré-jogo em histórico para slope
        elo_hist[a].append(elo[a])
        elo_hist[b].append(elo[b])

        # computa percentuais do jogo atual para alimentar deques
        def push_stats(pid, svpt, first_in, ace, dfault, first_won, second_won, bp_saved, bp_faced):
            # taxas defensivas/ofensivas (mesmas definições do seu pipeline)
            if svpt and svpt != 0:
                k_hist[pid]["p_ace"]     # garante estrutura
                for K in Ks:
                    k_hist[pid]["p_ace"][K].append(100 * (ace / svpt) if svpt else np.nan)
                    k_hist[pid]["p_df"][K].append(100 * (dfault / svpt) if svpt else np.nan)
                    k_hist[pid]["p_1stIn"][K].append(100 * (first_in / svpt) if svpt else np.nan)

            if first_in and first_in != 0:
                for K in Ks:
                    k_hist[pid]["p_1stWon"][K].append(100 * (first_won / first_in))

            if svpt and first_in is not None and (svpt - first_in) != 0:
                for K in Ks:
                    k_hist[pid]["p_2ndWon"][K].append(100 * (second_won / (svpt - first_in)))

            if bp_faced and bp_faced != 0:
                for K in Ks:
                    k_hist[pid]["p_bpSaved"][K].append(100 * (bp_saved / bp_faced))

        push_stats(
            a, r.player1_svpt, r.player1_1stIn, r.player1_ace, r.player1_df,
            r.player1_1stWon, r.player1_2ndWon, r.player1_bpSaved, r.player1_bpFaced
        )
        push_stats(
            b, r.player2_svpt, r.player2_1stIn, r.player2_ace, r.player2_df,
            r.player2_1stWon, r.player2_2ndWon, r.player2_bpSaved, r.player2_bpFaced
        )

        # atualiza Elo pós-jogo
        K = K_BASE * event_importance(getattr(r, "tourney_level", "A"), getattr(r, "round", ""))
        K *= sets_factor(getattr(r, "best_of", 3))
        Ea = expected_score(elo[a], elo[b])
        Eb = 1.0 - Ea
        elo[a] = elo[a] + K * (1.0 - Ea)
        elo[b] = elo[b] + K * (0.0 - Eb)

    state = {
        "elo": elo,
        "elo_hist": elo_hist,
        "last_rank": last_rank,
        "last_pts": last_pts,
        "last_age": last_age,
        "last_ht": last_ht,
        "h2h": h2h,
        "h2h_surface": h2h_surface,
        "k_hist": k_hist,
        "Ks": Ks,
        "METRICS": METRICS
    }
    return state

# ==============================
# 2) Geração de 1 linha de features para X vs Y (pré-jogo)
# ==============================

def _mean_k(player_id, metric, K, state):
    dq = state["k_hist"][player_id][metric][K]
    return safe_mean(dq)

def _elo_slope(player_id, K, state):
    hist = state["elo_hist"][player_id]
    # pega só os últimos K pontos
    if len(hist) == 0:
        return np.nan
    seq = hist[-K:] if len(hist) >= K else hist
    return rolling_slope(seq)

def make_feature_row(player1_id, player2_id, context, state):
    """
    Retorna um DataFrame (1 linha) com as MESMAS colunas do seu df_final.
    context: dict com chaves mínimas:
      - 'tourney_date' (datetime ou str yyyy-mm-dd)
      - 'surface' (ex.: 'Hard','Clay','Grass','Carpet')
      - 'round' (ex.: 'R32','QF','SF','F' etc.)
      - 'tourney_level' (ex.: 'A','M','G','C')
      - 'best_of' (3 ou 5)
      - 'draw_size' (int)
    IMPORTANTE: o "state" já deve estar construído APENAS com jogos ANTERIORES a essa data.
    """
    sfc = context.get("surface", None)

    # Diferenciais "básicos"
    atp_points_diff = (state["last_pts"].get(player1_id, np.nan)
                       - state["last_pts"].get(player2_id, np.nan))
    atp_rank_diff   = (state["last_rank"].get(player1_id, np.nan)
                       - state["last_rank"].get(player2_id, np.nan))
    age_diff        = (state["last_age"].get(player1_id, np.nan)
                       - state["last_age"].get(player2_id, np.nan))
    ht_diff         = (state["last_ht"].get(player1_id, np.nan)
                       - state["last_ht"].get(player2_id, np.nan))
    elo_diff        = (state["elo"].get(player1_id, 1500.0)
                       - state["elo"].get(player2_id, 1500.0))

    h2h_diff = state["h2h"].get((player1_id, player2_id), 0) - state["h2h"].get((player2_id, player1_id), 0)
    if sfc is not None:
        h2h_surf_diff = state["h2h_surface"][sfc].get((player1_id, player2_id), 0) - \
                        state["h2h_surface"][sfc].get((player2_id, player1_id), 0)
    else:
        h2h_surf_diff = 0

    # janelas
    Ks = state["Ks"]

    vals = {
        "player1_id": player1_id,
        "player2_id": player2_id,
        "best_of": context.get("best_of", 3),
        "draw_size": context.get("draw_size", np.nan),
        "atp_points_differential": atp_points_diff,
        "atp_rank_differential": atp_rank_diff,
        "age_differential": age_diff,
        "ht_differential": ht_diff,
        "elo_differential": elo_diff,
        "h2h_differential": h2h_diff,
        "h2h_surface_differential": h2h_surf_diff,
    }

    # métricas percentuais por K (sempre P1 - P2)
    # nomes precisam casar com o seu df_final
    name_map = {
        "p_ace": "p_ace_last{K}_differential",
        "p_df": "p_df_last{K}_differential",
        "p_1stIn": "p_1st_in_last{K}_differential",
        "p_1stWon": "p_1st_won_last{K}_differential",
        "p_2ndWon": "p_2nd_won_last{K}_differential",
        "p_bpSaved": "p_bp_saved_last{K}_differential",
    }
    for K in Ks:
        for metric, out_pat in name_map.items():
            m1 = _mean_k(player1_id, metric, K, state)
            m2 = _mean_k(player2_id, metric, K, state)
            vals[out_pat.format(K=K)] = (m1 - m2) if not (np.isnan(m1) and np.isnan(m2)) else np.nan
        # gradiente de Elo (slope) por K
        vals[f"elo_gradient_{K}_differential"] = _elo_slope(player1_id, K, state) - _elo_slope(player2_id, K, state)

    # ordena colunas exatamente como no seu df_final
    final_cols = [
        'player1_id', 'player2_id', 'best_of', 'draw_size',
        'atp_points_differential','atp_rank_differential','age_differential','ht_differential',
        'elo_differential','h2h_differential','h2h_surface_differential',
        'p_ace_last3_differential','p_df_last3_differential','p_1st_in_last3_differential',
        'p_1st_won_last3_differential','p_2nd_won_last3_differential','p_bp_saved_last3_differential',
        'elo_gradient_3_differential',
        'p_ace_last5_differential','p_df_last5_differential','p_1st_in_last5_differential',
        'p_1st_won_last5_differential','p_2nd_won_last5_differential','p_bp_saved_last5_differential',
        'elo_gradient_5_differential',
        'p_ace_last10_differential','p_df_last10_differential','p_1st_in_last10_differential',
        'p_1st_won_last10_differential','p_2nd_won_last10_differential','p_bp_saved_last10_differential',
        'elo_gradient_10_differential',
        'p_ace_last20_differential','p_df_last20_differential','p_1st_in_last20_differential',
        'p_1st_won_last20_differential','p_2nd_won_last20_differential','p_bp_saved_last20_differential',
        'elo_gradient_20_differential',
        'p_ace_last50_differential','p_df_last50_differential','p_1st_in_last50_differential',
        'p_1st_won_last50_differential','p_2nd_won_last50_differential','p_bp_saved_last50_differential',
        'elo_gradient_50_differential',
        'p_ace_last100_differential','p_df_last100_differential','p_1st_in_last100_differential',
        'p_1st_won_last100_differential','p_2nd_won_last100_differential','p_bp_saved_last100_differential',
        'elo_gradient_100_differential',
    ]
    row = pd.DataFrame([vals], columns=final_cols)
    return row