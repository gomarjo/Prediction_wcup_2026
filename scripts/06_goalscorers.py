# -*- coding: utf-8 -*-
"""
Modelo de goleadores (anytime scorer) — Copa del Mundo 2026.

Reparte el xG consenso del equipo (Dixon-Coles + XGBoost, neutral=True) entre
sus anotadores recientes segun su cuota historica de goles (ultimos 2 anios,
con decaimiento temporal de semivida 1 anio, excluyendo autogoles).

Probabilidades por jugador (Poisson con xg_j = xG_equipo * share_j):
    Anota (anytime): 1 - exp(-xg_j)
    2 o mas goles:   1 - exp(-xg_j) - xg_j * exp(-xg_j)

Salidas: output/partidoN/partidoN_goleadores.png (top 10 de ambos equipos)
y resumen en consola (top 5 por equipo).

Limitaciones: el dataset no tiene convocatorias ni alineaciones — un jugador
lesionado o no convocado puede aparecer, y uno nuevo sin goles no aparecera.
"""
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from matches_config import MATCHES  # noqa: E402

WINDOW_YEARS = 2
HALF_LIFE_DAYS = 365
DECAY_LAMBDA = np.log(2) / HALF_LIFE_DAYS

# ---------------------------------------------------------------- estilo
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
COLOR_HOME = "#2a78d6"   # azul (mismo par que el grafico xG del script 04)
COLOR_AWAY = "#eb6834"   # naranja

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------------------- modelo
def load_recent_goals():
    df = pd.read_csv(BASE_DIR / "data" / "goalscorers.csv", parse_dates=["date"])
    df = df.dropna(subset=["scorer"])
    df = df[~df["own_goal"].astype(bool)]  # autogoles: el scorer es rival
    today = pd.Timestamp.today()
    df = df[(df["date"] >= today - pd.DateOffset(years=WINDOW_YEARS)) &
            (df["date"] <= today)].copy()
    df["weight"] = np.exp(-DECAY_LAMBDA * (today - df["date"]).dt.days)
    return df


def team_scorer_shares(goals_df, team):
    """Retorna DataFrame por jugador: share (pesos normalizados), goles crudos
    y si es el lanzador de penales del equipo en la ventana."""
    tg = goals_df[goals_df["team"] == team]
    if tg.empty:
        return pd.DataFrame(columns=["scorer", "share", "goals", "is_pen_taker"])
    by_player = tg.groupby("scorer").agg(
        weight=("weight", "sum"), goals=("scorer", "size")).reset_index()
    by_player["share"] = by_player["weight"] / by_player["weight"].sum()

    pens = tg[tg["penalty"].astype(bool)].groupby("scorer")["weight"].sum()
    pen_taker = pens.idxmax() if not pens.empty else None
    by_player["is_pen_taker"] = by_player["scorer"] == pen_taker
    return by_player.sort_values("share", ascending=False).reset_index(drop=True)


def scorer_probabilities(shares, team_xg):
    df = shares.copy()
    df["player_xg"] = team_xg * df["share"]
    df["p_anytime"] = 1 - np.exp(-df["player_xg"])
    df["p_two_plus"] = df["p_anytime"] - df["player_xg"] * np.exp(-df["player_xg"])
    return df


# ---------------------------------------------------------------- grafico
def plot_scorers(match_title, home_es, away_es, home_df, away_df, path):
    home_df = home_df.assign(team_label=home_es, color=COLOR_HOME)
    away_df = away_df.assign(team_label=away_es, color=COLOR_AWAY)
    top = (pd.concat([home_df, away_df])
           .sort_values("p_anytime", ascending=False).head(10)
           .iloc[::-1].reset_index(drop=True))  # invertido: mayor arriba

    fig, ax = plt.subplots(figsize=(9.5, 0.52 * len(top) + 1.9), dpi=150)
    y = np.arange(len(top))
    ax.barh(y, top["p_anytime"] * 100, height=0.6, color=top["color"], zorder=3)
    for yi, row in top.iterrows():
        ax.text(row["p_anytime"] * 100 + 0.8, yi,
                f"{row['p_anytime'] * 100:.1f}%  (2+: {row['p_two_plus'] * 100:.1f}%)",
                va="center", ha="left", fontsize=9, color=INK)

    labels = [f"{r['scorer']}{' ✱' if r['is_pen_taker'] else ''}"
              for _, r in top.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.set_xlim(0, top["p_anytime"].max() * 100 * 1.38)
    ax.set_xlabel("Probabilidad de anotar en el partido (%)", fontsize=10, color=INK_2)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)

    legend_items = [plt.Rectangle((0, 0), 1, 1, color=COLOR_HOME),
                    plt.Rectangle((0, 0), 1, 1, color=COLOR_AWAY)]
    ax.legend(legend_items, [home_es, away_es], loc="lower right",
              bbox_to_anchor=(1.0, 1.0), ncols=2, frameon=False,
              fontsize=9.5, labelcolor=INK_2)

    ax.set_title(f"{match_title} — Goleadores Más Probables",
                 fontsize=13, fontweight="bold", loc="left", pad=34)
    fig.text(0.01, 0.005, "✱ = lanzador de penales habitual · Basado en goles de los "
             "últimos 2 años; no considera convocatorias ni alineaciones",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- principal
def main():
    with open(BASE_DIR / "models" / "dixon_coles.pkl", "rb") as f:
        dc_model = pickle.load(f)
    with open(BASE_DIR / "models" / "xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    goals_df = load_recent_goals()
    print(f"Goles en la ventana ({WINDOW_YEARS} años, sin autogoles): {len(goals_df):,}")

    line = "═" * 60
    print()
    print(line)
    print("GOLEADORES MÁS PROBABLES — Copa del Mundo 2026")
    print(line)

    for num, home_es, away_es, home_ds, away_ds in MATCHES:
        title = f"{home_es} vs {away_es}"
        dc = dc_model.predict(home_ds, away_ds, neutral=True)
        xgb = xgb_model.predict(home_ds, away_ds, neutral=True)
        xg_consensus = {
            home_ds: (dc["home_xg"] + xgb["home_xg"]) / 2,
            away_ds: (dc["away_xg"] + xgb["away_xg"]) / 2,
        }

        team_probs = {}
        print(f"\nPartido {num}: {title}")
        for team_ds, team_es in ((home_ds, home_es), (away_ds, away_es)):
            shares = team_scorer_shares(goals_df, team_ds)
            if shares.empty:
                print(f"  AVISO: sin goles registrados de {team_es} en la ventana.")
                team_probs[team_ds] = scorer_probabilities(shares, 0.0)
                continue
            probs = scorer_probabilities(shares, xg_consensus[team_ds])
            team_probs[team_ds] = probs
            print(f"\n  {team_es} (xG consenso: {xg_consensus[team_ds]:.2f}) — top 5:")
            print(f"    {'Jugador':<28}{'Share':>7}{'xG':>7}{'Anota':>8}{'2+':>7}")
            for _, r in probs.head(5).iterrows():
                pen = " ✱" if r["is_pen_taker"] else ""
                print(f"    {r['scorer'] + pen:<28}"
                      f"{r['share'] * 100:>6.1f}%"
                      f"{r['player_xg']:>7.2f}"
                      f"{r['p_anytime'] * 100:>7.1f}%"
                      f"{r['p_two_plus'] * 100:>6.1f}%")

        out_dir = BASE_DIR / "output" / f"partido{num}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / f"partido{num}_goleadores.png"
        plot_scorers(title, home_es, away_es,
                     team_probs[home_ds], team_probs[away_ds], out_path)
        print(f"\n  Gráfico guardado en {out_path}")

    print()
    print(line)
    print("✱ = lanzador de penales habitual del equipo (últimos 2 años)")
    print("⚠️  Limitaciones: el modelo reparte el xG del equipo según los goles")
    print("de los últimos 2 años. No considera convocatorias, alineaciones ni")
    print("lesiones: verifica que el jugador esté convocado antes de usar el dato.")
    print(line)


if __name__ == "__main__":
    main()
