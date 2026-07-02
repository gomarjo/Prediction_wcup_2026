# -*- coding: utf-8 -*-
"""
FASES 5-8 - Predicciones de la Copa del Mundo 2026 y visualizaciones.

Genera para cada partido (sede neutral, neutral=True):
    1. Ganador mas probable en 90 min (barras horizontales G/E/P, ambos modelos)
    2. Goles esperados xG (barras agrupadas + promedio historico de referencia)
    3. Top 10 marcadores mas probables (tabla visual)

Guarda los PNG en output/partidoN/ e imprime el resumen final en consola.
"""
import difflib
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import poisson

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from match_features import load_matches  # noqa: E402
from matches_config import MATCHES  # noqa: E402

# ---------------------------------------------------------------- estilo
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# G/E/P: verde / gris / rojo — tono claro = Dixon-Coles, oscuro = XGBoost
COLOR_GEP = {
    "G": {"dc": "#8fce8f", "xgb": "#008300"},
    "E": {"dc": "#c3c2b7", "xgb": "#6b6a65"},
    "P": {"dc": "#f0a3a2", "xgb": "#d03b3b"},
}
COLOR_HOME = {"dc": "#86b6ef", "xgb": "#1c5cab"}   # azul claro/oscuro
COLOR_AWAY = {"dc": "#f3b391", "xgb": "#d95926"}   # naranja claro/oscuro
HILITE_BG = "#cde2fb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def resolve_team(name, known_teams, df):
    """Si el equipo no existe, avisa, muestra candidatos y usa el mas parecido."""
    if name in known_teams:
        return name
    print(f"AVISO: '{name}' no existe en el dataset. Buscando nombre parecido...")
    print(df[df["home_team"].str.contains(name, case=False)].head())
    closest = difflib.get_close_matches(name, known_teams, n=1, cutoff=0.4)
    if not closest:
        raise KeyError(f"No se encontro ningun equipo parecido a '{name}'")
    print(f"  Usando el mas parecido: '{closest[0]}'")
    return closest[0]


def poisson_score_matrix(home_xg, away_xg, max_goals=10):
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, home_xg), poisson.pmf(goals, away_xg))
    return matrix / matrix.sum()


def style_axes(ax):
    ax.tick_params(length=0)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
        ax.spines[spine].set_linewidth(1)


# ---------------------------------------------------------------- graficos
def plot_winner(match_title, home_es, away_es, dc, xgb, path):
    labels = [f"G — {home_es}", "E — Empate", f"P — {away_es}"]
    keys = ["G", "E", "P"]
    dc_vals = [dc["home_win_prob"], dc["draw_prob"], dc["away_win_prob"]]
    xgb_vals = [xgb["home_win_prob"], xgb["draw_prob"], xgb["away_win_prob"]]
    consensus = [(d + x) / 2 for d, x in zip(dc_vals, xgb_vals)]
    best = int(np.argmax(consensus))

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    y = np.arange(3)[::-1]  # G arriba
    h = 0.32
    for i, key in enumerate(keys):
        highlight = dict(edgecolor=INK, linewidth=1.6) if i == best else \
            dict(edgecolor="none", linewidth=0)
        b_dc = ax.barh(y[i] + h / 2 + 0.02, dc_vals[i] * 100, height=h,
                       color=COLOR_GEP[key]["dc"], zorder=3, **highlight)
        b_xgb = ax.barh(y[i] - h / 2 - 0.02, xgb_vals[i] * 100, height=h,
                        color=COLOR_GEP[key]["xgb"], zorder=3, **highlight)
        for bars, val in ((b_dc, dc_vals[i]), (b_xgb, xgb_vals[i])):
            bar = bars[0]
            ax.text(bar.get_width() + 1.2, bar.get_y() + bar.get_height() / 2,
                    f"{val * 100:.1f}%", va="center", ha="left",
                    fontsize=10, color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11, color=INK)
    ax.set_xlim(0, max(max(dc_vals), max(xgb_vals)) * 100 * 1.22)
    ax.set_xlabel("Probabilidad (%)", fontsize=10)
    ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    legend_items = [
        plt.Rectangle((0, 0), 1, 1, color="#c3c2b7"),
        plt.Rectangle((0, 0), 1, 1, color="#6b6a65"),
    ]
    ax.legend(legend_items, ["Dixon-Coles (tono claro)", "XGBoost (tono oscuro)"],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2,
              frameon=False, fontsize=9, labelcolor=INK_2)

    ax.set_title(f"{match_title} — Resultado en 90 min",
                 fontsize=13, fontweight="bold", loc="left", pad=36)
    fig.text(0.01, 0.005, "Borde negro = resultado más probable según consenso "
             "(promedio de ambos modelos)", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_xg(match_title, home_es, away_es, dc, xgb, hist_avg, path):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    x = np.array([0, 1])  # Dixon-Coles, XGBoost
    w = 0.3
    home_vals = [dc["home_xg"], xgb["home_xg"]]
    away_vals = [dc["away_xg"], xgb["away_xg"]]

    for xi, (hv, av, tone) in enumerate(zip(home_vals, away_vals, ("dc", "xgb"))):
        bh = ax.bar(xi - w / 2 - 0.015, hv, width=w, color=COLOR_HOME[tone], zorder=3)
        ba = ax.bar(xi + w / 2 + 0.015, av, width=w, color=COLOR_AWAY[tone], zorder=3)
        for bars, val in ((bh, hv), (ba, av)):
            bar = bars[0]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.04,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=10, color=INK)

    ax.axhline(hist_avg, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.4, zorder=2)
    ax.text(1.48, hist_avg + 0.03, f"Promedio histórico\n{hist_avg:.2f} goles/equipo",
            fontsize=8.5, color=INK_2, ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(["Dixon-Coles", "XGBoost"], fontsize=11, color=INK)
    ax.set_ylabel("Goles esperados (xG)", fontsize=10)
    ax.set_ylim(0, max(home_vals + away_vals + [hist_avg]) * 1.28)
    ax.set_xlim(-0.55, 1.55)
    ax.yaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)

    legend_items = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_HOME["xgb"]),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_AWAY["xgb"]),
    ]
    ax.legend(legend_items, [home_es, away_es], loc="upper right",
              frameon=False, fontsize=9.5, labelcolor=INK_2)

    ax.set_title(f"{match_title} — Goles Esperados (xG)",
                 fontsize=13, fontweight="bold", loc="left", pad=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def top10_scorelines(dc_matrix, xgb_matrix):
    """Combina ambas matrices y retorna las 10 filas [(i,j), p_dc, p_xgb, p_con]."""
    rows = []
    for i in range(7):
        for j in range(7):
            p_dc = dc_matrix.get((i, j), 0.0)
            p_xgb = float(xgb_matrix[i, j])
            rows.append(((i, j), p_dc, p_xgb, (p_dc + p_xgb) / 2))
    rows.sort(key=lambda r: r[3], reverse=True)
    return rows[:10]


def plot_top10(match_title, home_es, away_es, top10, path):
    col_labels = ["Posición", "Marcador", "Prob.\nDixon-Coles",
                  "Prob.\nXGBoost", "Prob.\nConsenso"]
    cell_text = []
    for pos, ((i, j), p_dc, p_xgb, p_con) in enumerate(top10, start=1):
        cell_text.append([
            str(pos),
            f"{home_es} {i} - {j} {away_es}",
            f"{p_dc * 100:.1f}%",
            f"{p_xgb * 100:.1f}%",
            f"{p_con * 100:.1f}%",
        ])

    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center",
                     cellLoc="center", colWidths=[0.11, 0.41, 0.16, 0.16, 0.16])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.75)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(1)
        if row == 0:
            cell.set_facecolor(SURFACE)
            cell.set_text_props(color=INK_2, fontweight="bold", fontsize=9)
            cell.set_height(0.14)
        elif row == 1:  # marcador mas probable
            cell.set_facecolor(HILITE_BG)
            cell.set_text_props(color=INK, fontweight="bold")
        else:
            cell.set_facecolor(SURFACE)
            cell.set_text_props(color=INK)

    ax.set_title(f"{match_title} — Top 10 Marcadores Más Probables",
                 fontsize=13, fontweight="bold", loc="left", pad=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- principal
def main():
    with open(BASE_DIR / "models" / "dixon_coles.pkl", "rb") as f:
        dc_model = pickle.load(f)
    with open(BASE_DIR / "models" / "xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    df = load_matches()
    # Promedio historico de goles por equipo y partido (referencia del grafico xG)
    hist_avg = float((df["home_score"] + df["away_score"]).mean() / 2)
    known_teams = set(dc_model.teams()) & set(xgb_model.teams())

    summary = []
    for num, home_es, away_es, home_ds, away_ds in MATCHES:
        home = resolve_team(home_ds, known_teams, df)
        away = resolve_team(away_ds, known_teams, df)
        title = f"{home_es} vs {away_es}"
        print(f"Generando predicciones y graficos: Partido {num} — {title}")

        # Sede neutral: Copa del Mundo 2026 (Mexico/Canada/EE.UU.)
        dc = dc_model.predict(home, away, neutral=True)
        xgb = xgb_model.predict(home, away, neutral=True)
        xgb_matrix = poisson_score_matrix(xgb["home_xg"], xgb["away_xg"])
        top10 = top10_scorelines(dc["score_matrix"], xgb_matrix)

        out_dir = BASE_DIR / "output" / f"partido{num}"
        os.makedirs(out_dir, exist_ok=True)
        plot_winner(title, home_es, away_es, dc, xgb,
                    out_dir / f"partido{num}_ganador.png")
        plot_xg(title, home_es, away_es, dc, xgb, hist_avg,
                out_dir / f"partido{num}_xg.png")
        plot_top10(title, home_es, away_es, top10,
                   out_dir / f"partido{num}_top10.png")

        (i, j), _, _, _ = top10[0]
        summary.append((num, title, home_es, away_es, dc, xgb, (i, j)))

    # FASE 8 — Resumen final
    line = "═" * 50
    print()
    print(line)
    print("PREDICCIONES — Copa del Mundo 2026")
    print(line)
    for num, title, home_es, away_es, dc, xgb, (i, j) in summary:
        print()
        print(f"Partido {num}: {title}")
        for tag, p in (("Dixon-Coles", dc), ("XGBoost    ", xgb)):
            print(f"  {tag} → G: {p['home_win_prob'] * 100:.1f}% | "
                  f"E: {p['draw_prob'] * 100:.1f}% | "
                  f"P: {p['away_win_prob'] * 100:.1f}% | "
                  f"xG: {p['home_xg']:.2f} - {p['away_xg']:.2f}")
        print(f"  Marcador más probable: {home_es} {i} - {j} {away_es}")
    print()
    print(line)


if __name__ == "__main__":
    main()
