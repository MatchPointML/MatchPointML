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
    df = all_matches.copy()
    df.columns = [c.replace('w_', 'player1_').replace('winner_', 'player1_')
                    .replace('l_', 'player2_').replace('loser_', 'player2_')
                  for c in df.columns]
    df = df.sort_values(["tourney_date", "match_num"], kind="mergesort").reset_index(drop=True)

    # Elo geral
    elo = defaultdict(lambda: 1500.0)
    elo_hist = defaultdict(list)

    # Elo por superfície
    surfaces_known = ["Hard","Clay","Grass","Carpet"]
    elo_surface = {s: defaultdict(lambda: 1500.0) for s in surfaces_known}
    elo_surface_hist = {s: defaultdict(list) for s in surfaces_known}

    # Últimos dados "simples"
    last_rank, last_pts, last_age, last_ht = {}, {}, {}, {}

    # H2H geral e por superfície
    h2h = defaultdict(int)
    h2h_surface = defaultdict(lambda: defaultdict(int))  # surface -> (a,b) -> vitórias

    # Deques de estatísticas por janela K
    Ks = [3,5,10,20,50,100]
    k_hist = defaultdict(lambda: defaultdict(lambda: {
        3: deque(maxlen=3), 5: deque(maxlen=5), 10: deque(maxlen=10),
        20: deque(maxlen=20), 50: deque(maxlen=50), 100: deque(maxlen=100)
    }))
    METRICS = ("p_ace","p_df","p_1stIn","p_1stWon","p_2ndWon","p_bpSaved")

    # Novos: histórico de vitórias (geral e por superfície)
    wins_hist = defaultdict(lambda: {
        3: deque(maxlen=3), 5: deque(maxlen=5), 10: deque(maxlen=10),
        20: deque(maxlen=20), 50: deque(maxlen=50), 100: deque(maxlen=100)
    })
    wins_surface_hist = {s: defaultdict(lambda: {
        3: deque(maxlen=3), 5: deque(maxlen=5), 10: deque(maxlen=10),
        20: deque(maxlen=20), 50: deque(maxlen=50), 100: deque(maxlen=100)
    }) for s in surfaces_known}

    for r in df.itertuples(index=False):
        a = r.player1_id  # winner
        b = r.player2_id  # loser
        surf = getattr(r, "surface", None)
        if pd.isna(surf) or surf not in surfaces_known:
            surf = None  # ignora superfícies fora do set conhecido

        # last known antes de atualizar
        last_rank[a] = getattr(r, "player1_rank", np.nan)
        last_rank[b] = getattr(r, "player2_rank", np.nan)
        last_pts[a]  = getattr(r, "player1_rank_points", np.nan)
        last_pts[b]  = getattr(r, "player2_rank_points", np.nan)
        last_age[a]  = getattr(r, "player1_age", np.nan)
        last_age[b]  = getattr(r, "player2_age", np.nan)
        last_ht[a]   = getattr(r, "player1_ht", np.nan)
        last_ht[b]   = getattr(r, "player2_ht", np.nan)

        # H2H
        h2h[(a,b)] += 1
        if surf:
            h2h_surface[surf][(a,b)] += 1

        # Elo pré-jogo para slope
        elo_hist[a].append(elo[a]); elo_hist[b].append(elo[b])
        if surf:
            elo_surface_hist[surf][a].append(elo_surface[surf][a])
            elo_surface_hist[surf][b].append(elo_surface[surf][b])

        # Alimenta deques de métricas
        def push_stats(pid, svpt, first_in, ace, dfault, first_won, second_won, bp_saved, bp_faced):
            if svpt and svpt != 0:
                k_hist[pid]["p_ace"]  # garante estrutura
                for K in Ks:
                    k_hist[pid]["p_ace"][K].append(100 * (ace / svpt))
                    k_hist[pid]["p_df"][K].append(100 * (dfault / svpt))
                    k_hist[pid]["p_1stIn"][K].append(100 * (first_in / svpt))
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

        # Atualiza vitórias recentes (1 para vencedor, 0 para perdedor)
        for K in Ks:
            wins_hist[a][K].append(1); wins_hist[b][K].append(0)
            if surf:
                wins_surface_hist[surf][a][K].append(1)
                wins_surface_hist[surf][b][K].append(0)

        # Atualiza Elo geral
        Kf = K_BASE * event_importance(getattr(r, "tourney_level", "A"), getattr(r, "round", ""))
        Kf *= sets_factor(getattr(r, "best_of", 3))
        Ea = expected_score(elo[a], elo[b]); Eb = 1.0 - Ea
        elo[a] = elo[a] + Kf * (1.0 - Ea)
        elo[b] = elo[b] + Kf * (0.0 - Eb)

        # Atualiza Elo por superfície (mesmo Kf)
        if surf:
            Ea_s = expected_score(elo_surface[surf][a], elo_surface[surf][b]); Eb_s = 1.0 - Ea_s
            elo_surface[surf][a] = elo_surface[surf][a] + Kf * (1.0 - Ea_s)
            elo_surface[surf][b] = elo_surface[surf][b] + Kf * (0.0 - Eb_s)

    state = {
        "elo": elo,
        "elo_hist": elo_hist,
        "elo_surface": elo_surface,
        "elo_surface_hist": elo_surface_hist,
        "last_rank": last_rank,
        "last_pts": last_pts,
        "last_age": last_age,
        "last_ht": last_ht,
        "h2h": h2h,
        "h2h_surface": h2h_surface,
        "k_hist": k_hist,
        "Ks": Ks,
        "METRICS": METRICS,
        "wins_hist": wins_hist,
        "wins_surface_hist": wins_surface_hist,
        "surfaces_known": surfaces_known
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

def _wins_count(player_id, K, state):
    dq = state["wins_hist"][player_id][K]
    return int(sum(dq)) if len(dq) else 0

def _wins_rate(player_id, K, state):
    dq = state["wins_hist"][player_id][K]
    return float(np.mean(dq)) if len(dq) else np.nan

def _wins_count_surface(player_id, K, sfc, state):
    if not sfc or sfc not in state["surfaces_known"]:
        return np.nan
    dq = state["wins_surface_hist"][sfc][player_id][K]
    return int(sum(dq)) if len(dq) else 0

def _wins_rate_surface(player_id, K, sfc, state):
    if not sfc or sfc not in state["surfaces_known"]:
        return np.nan
    dq = state["wins_surface_hist"][sfc][player_id][K]
    return float(np.mean(dq)) if len(dq) else np.nan

def _elo_surface_value(player_id, sfc, state):
    if not sfc or sfc not in state["surfaces_known"]:
        return np.nan
    return state["elo_surface"][sfc].get(player_id, 1500.0)


def make_feature_row(player1_id, player2_id, context, state):
    sfc = context.get("surface", None)

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

    # NOVO: diferencial de Elo na superfície do contexto
    elo_surface_diff = (_elo_surface_value(player1_id, sfc, state)
                        - _elo_surface_value(player2_id, sfc, state))

    h2h_diff = state["h2h"].get((player1_id, player2_id), 0) - state["h2h"].get((player2_id, player1_id), 0)
    if sfc is not None:
        h2h_surf_diff = state["h2h_surface"][sfc].get((player1_id, player2_id), 0) - \
                        state["h2h_surface"][sfc].get((player2_id, player1_id), 0)
    else:
        h2h_surf_diff = 0

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
        "elo_surface_differential": elo_surface_diff,   # NOVO
        "h2h_differential": h2h_diff,
        "h2h_surface_differential": h2h_surf_diff,
    }

    name_map = {
        "p_ace": "p_ace_last{K}_differential",
        "p_df": "p_df_last{K}_differential",
        "p_1stIn": "p_1st_in_last{K}_differential",
        "p_1stWon": "p_1st_won_last{K}_differential",
        "p_2ndWon": "p_2nd_won_last{K}_differential",
        "p_bpSaved": "p_bp_saved_last{K}_differential",
    }
    for K in Ks:
        # métricas percentuais
        for metric, out_pat in name_map.items():
            m1 = _mean_k(player1_id, metric, K, state)
            m2 = _mean_k(player2_id, metric, K, state)
            vals[out_pat.format(K=K)] = (m1 - m2) if not (np.isnan(m1) and np.isnan(m2)) else np.nan
        # gradiente de Elo geral
        vals[f"elo_gradient_{K}_differential"] = _elo_slope(player1_id, K, state) - _elo_slope(player2_id, K, state)

        # NOVO: vitórias recentes (contagem) e taxa (geral)
        vals[f"wins_last{K}_differential"] = _wins_count(player1_id, K, state) - _wins_count(player2_id, K, state)
        vals[f"wins_last{K}_differential"] = _wins_rate(player1_id, K, state) - _wins_rate(player2_id, K, state)

    # COLUNAS: mantenha as antigas e acrescente as novas no final
    final_cols = [
        'player1_id', 'player2_id', 'best_of', 'draw_size',
        'atp_points_differential','atp_rank_differential','age_differential','ht_differential',
        'elo_differential','elo_surface_differential',
        'h2h_differential','h2h_surface_differential',

        'p_ace_last3_differential','p_df_last3_differential','p_1st_in_last3_differential',
        'p_1st_won_last3_differential','p_2nd_won_last3_differential','p_bp_saved_last3_differential',
        'elo_gradient_3_differential','elo_surface_gradient_3_differential',
        'wins_last3_differential',

        'p_ace_last5_differential','p_df_last5_differential','p_1st_in_last5_differential',
        'p_1st_won_last5_differential','p_2nd_won_last5_differential','p_bp_saved_last5_differential',
        'elo_gradient_5_differential','elo_surface_gradient_5_differential',
        'wins_last5_differential',

        'p_ace_last10_differential','p_df_last10_differential','p_1st_in_last10_differential',
        'p_1st_won_last10_differential','p_2nd_won_last10_differential','p_bp_saved_last10_differential',
        'elo_gradient_10_differential','elo_surface_gradient_10_differential',
        'wins_last10_differential',

        'p_ace_last20_differential','p_df_last20_differential','p_1st_in_last20_differential',
        'p_1st_won_last20_differential','p_2nd_won_last20_differential','p_bp_saved_last20_differential',
        'elo_gradient_20_differential','elo_surface_gradient_20_differential',
        'wins_last20_differential',
    ]
    row = pd.DataFrame([vals], columns=final_cols)
    return row


['best_of', 'draw_size', 'atp_points_differential', 'atp_rank_differential', 'age_differential', 'ht_differential', 'elo_differential', 'elo_surface_differential', 'h2h_differential', 'h2h_surface_differential', 'p_ace_last3_differential', 'p_df_last3_differential', 'p_1st_in_last3_differential', 'p_1st_won_last3_differential', 'p_2nd_won_last3_differential', 'p_bp_saved_last3_differential', 'elo_gradient_3_differential', 'elo_surface_gradient_3_differential', 'wins_last3_differential', 'p_ace_last5_differential', 'p_df_last5_differential', 'p_1st_in_last5_differential', 'p_1st_won_last5_differential', 'p_2nd_won_last5_differential', 'p_bp_saved_last5_differential', 'elo_gradient_5_differential', 'elo_surface_gradient_5_differential', 'wins_last5_differential', 'p_ace_last10_differential', 'p_df_last10_differential', 'p_1st_in_last10_differential', 'p_1st_won_last10_differential', 'p_2nd_won_last10_differential', 'p_bp_saved_last10_differential', 'elo_gradient_10_differential', 'elo_surface_gradient_10_differential', 'wins_last10_differential', 'p_ace_last20_differential', 'p_df_last20_differential', 'p_1st_in_last20_differential', 'p_1st_won_last20_differential', 'p_2nd_won_last20_differential', 'p_bp_saved_last20_differential', 'elo_gradient_20_differential', 'elo_surface_gradient_20_differential', 'wins_last20_differential']


['best_of', 'draw_size', 'atp_points_differential', 'atp_rank_differential', 'age_differential', 'ht_differential', 'elo_differential', 'elo_surface_differential', 'h2h_differential', 'h2h_surface_differential', 'p_ace_last3_differential', 'p_df_last3_differential', 'p_1st_in_last3_differential', 'p_1st_won_last3_differential', 'p_2nd_won_last3_differential', 'p_bp_saved_last3_differential', 'elo_gradient_3_differential', 'elo_surface_gradient_3_differential', 'wins_last3_differential', 'wins_last3_differential', 'p_ace_last5_differential', 'p_df_last5_differential', 'p_1st_in_last5_differential', 'p_1st_won_last5_differential', 'p_2nd_won_last5_differential', 'p_bp_saved_last5_differential', 'elo_gradient_5_differential', 'elo_surface_gradient_5_differential', 'wins_last5_differential', 'wins_last5_differential', 'p_ace_last10_differential', 'p_df_last10_differential', 'p_1st_in_last10_differential', 'p_1st_won_last10_differential', 'p_2nd_won_last10_differential', 'p_bp_saved_last10_differential', 'elo_gradient_10_differential', 'elo_surface_gradient_10_differential', 'wins_last10_differential', 'wins_last10_differential', 'p_ace_last20_differential', 'p_df_last20_differential', 'p_1st_in_last20_differential', 'p_1st_won_last20_differential', 'p_2nd_won_last20_differential', 'p_bp_saved_last20_differential', 'elo_gradient_20_differential', 'elo_surface_gradient_20_differential', 'wins_last20_differential', 'wins_last20_differential']



