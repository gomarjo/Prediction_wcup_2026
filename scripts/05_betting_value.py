# -*- coding: utf-8 -*-
"""
Analisis de valor en apuestas (EV) — Copa del Mundo 2026.

Compara las probabilidades consenso de los modelos (Dixon-Coles + XGBoost)
contra las cuotas decimales del mercado, calcula el valor esperado, el stake
sugerido con Kelly fraccionado (25%, cap 5% del bankroll) y genera una tabla
de picks (output/betting_picks.png) mas un resumen en consola.

Mercados: 1X2 (obligatorio) y, si se ingresan cuotas, BTTS, Mas/Menos 2.5
y resultado exacto (#1 del top 10). Deja en blanco una cuota opcional para
omitir ese mercado.
"""
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

# Consola Windows: evitar fallos con acentos/emoji
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from matches_config import MATCHES  # noqa: E402

KELLY_FRACTION = 0.25
STAKE_CAP = 0.05           # nunca mas del 5% del bankroll por pick
UNCERTAINTY_LIMIT = 0.15   # >15 pp de diferencia entre modelos => excluir

# ---------------------------------------------------------------- estilo
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
SIGNAL_STYLE = {
    "VALOR ALTO": {"emoji": "✅", "bg": "#c9e7c9"},
    "VALOR MEDIO": {"emoji": "⚠️", "bg": "#fce3a8"},
    "VALOR BAJO": {"emoji": "🔍", "bg": "#e3e2dc"},
    "SIN VALOR": {"emoji": "❌", "bg": "#f4c8c7"},
}


# ---------------------------------------------------------------- helpers
def ask_float(prompt, minimum=1.01, allow_blank=False):
    while True:
        raw = input(prompt).strip().replace(",", ".").replace("$", "")
        if not raw:
            if allow_blank:
                return None
            print("  Valor requerido. Intenta de nuevo.")
            continue
        try:
            value = float(raw)
        except ValueError:
            print("  Numero invalido. Ejemplo valido: 1.85")
            continue
        if value < minimum:
            print(f"  Debe ser >= {minimum}. Intenta de nuevo.")
            continue
        return value


def poisson_score_matrix(home_xg, away_xg, max_goals=10):
    goals = np.arange(max_goals + 1)
    matrix = np.outer(poisson.pmf(goals, home_xg), poisson.pmf(goals, away_xg))
    return matrix / matrix.sum()


def top_scoreline(dc_matrix, xgb_matrix):
    """Marcador (i,j) con mayor prob. consenso y sus probabilidades."""
    best = None
    for i in range(7):
        for j in range(7):
            p_dc = float(dc_matrix[i, j])
            p_xgb = float(xgb_matrix[i, j])
            p_con = (p_dc + p_xgb) / 2
            if best is None or p_con > best[3]:
                best = ((i, j), p_dc, p_xgb, p_con)
    return best


def classify(ev):
    if ev >= 0.05:
        return "VALOR ALTO"
    if ev >= 0.02:
        return "VALOR MEDIO"
    if ev >= 0.0:
        return "VALOR BAJO"
    return "SIN VALOR"


# ---------------------------------------------------------------- analisis
def build_pick(match_title, market, pick_label, odds, prob_model,
               prob_dc=None, prob_xgb=None):
    """Retorna un dict con EV, Kelly y senal; marca alta incertidumbre si
    ambos modelos opinan y difieren mas de UNCERTAINTY_LIMIT."""
    uncertainty = abs(prob_dc - prob_xgb) if (prob_dc is not None and prob_xgb is not None) else 0.0
    ev = prob_model * odds - 1
    kelly_full = (prob_model * odds - 1) / (odds - 1)
    kelly_25 = max(kelly_full, 0.0) * KELLY_FRACTION
    return {
        "partido": match_title,
        "mercado": market,
        "pick": pick_label,
        "cuota": odds,
        "prob_model": prob_model,
        "prob_bookie": 1 / odds,
        "ev": ev,
        "kelly": kelly_25,
        "senal": classify(ev),
        "uncertainty": uncertainty,
        "excluded": uncertainty > UNCERTAINTY_LIMIT,
    }


def collect_inputs_and_analyze(dc_model, xgb_model):
    print("═" * 50)
    print("ANALIZADOR DE VALOR — Copa del Mundo 2026")
    print("═" * 50)
    bankroll = ask_float("\n¿Cuál es tu bankroll disponible para esta jornada? $", minimum=1.0)

    picks = []
    for num, home_es, away_es, home_ds, away_ds in MATCHES:
        title = f"{home_es} vs {away_es}"
        dc = dc_model.predict(home_ds, away_ds, neutral=True)
        xgb = xgb_model.predict(home_ds, away_ds, neutral=True)
        dc_matrix = dc_model.score_probability_matrix(home_ds, away_ds, neutral=True)
        xgb_matrix = poisson_score_matrix(xgb["home_xg"], xgb["away_xg"])

        print(f"\nPartido {num}: {title}")
        print("  — Mercado 1X2 (obligatorio) —")
        odds_g = ask_float(f"  Cuota {home_es} (G):  ")
        odds_e = ask_float("  Cuota Empate (E):  ")
        odds_p = ask_float(f"  Cuota {away_es} (P): ")

        for key, label, odds in (
            ("home_win_prob", f"{home_es} (G)", odds_g),
            ("draw_prob", "Empate (E)", odds_e),
            ("away_win_prob", f"{away_es} (P)", odds_p),
        ):
            consensus = (dc[key] + xgb[key]) / 2
            picks.append(build_pick(title, "1X2", label, odds, consensus,
                                    prob_dc=dc[key], prob_xgb=xgb[key]))

        print("  — Mercados adicionales (deja en blanco para omitir) —")
        odds_btts = ask_float("  Cuota BTTS (ambos anotan - Sí): ", allow_blank=True)
        odds_over = ask_float("  Cuota Más de 2.5 goles:  ", allow_blank=True)
        odds_under = ask_float("  Cuota Menos de 2.5 goles: ", allow_blank=True)
        odds_exact = ask_float("  Cuota resultado exacto (#1 del top 10): ", allow_blank=True)

        # BTTS y Over/Under con la score_matrix de Dixon-Coles
        p_home_scores = 1 - dc_matrix[0, :].sum()
        p_away_scores = 1 - dc_matrix[:, 0].sum()
        prob_btts = p_home_scores * p_away_scores
        idx_h, idx_a = np.indices(dc_matrix.shape)
        prob_over25 = float(dc_matrix[(idx_h + idx_a) > 2.5].sum())
        prob_under25 = 1 - prob_over25

        if odds_btts:
            picks.append(build_pick(title, "BTTS", "Ambos anotan: Sí",
                                    odds_btts, prob_btts))
        if odds_over:
            picks.append(build_pick(title, "Goles 2.5", "Más de 2.5",
                                    odds_over, prob_over25))
        if odds_under:
            picks.append(build_pick(title, "Goles 2.5", "Menos de 2.5",
                                    odds_under, prob_under25))
        if odds_exact:
            (gi, gj), p_dc, p_xgb, p_con = top_scoreline(dc_matrix, xgb_matrix)
            picks.append(build_pick(title, "Exacto",
                                    f"{home_es} {gi} - {gj} {away_es}",
                                    odds_exact, p_con,
                                    prob_dc=p_dc, prob_xgb=p_xgb))

    # Stake con cap del 5% del bankroll
    for p in picks:
        p["stake"] = 0.0 if p["excluded"] else round(
            bankroll * min(p["kelly"], STAKE_CAP), 2)
    return bankroll, picks


# ---------------------------------------------------------------- outputs
def save_picks_table(picks, path):
    rows = [p for p in picks if p["ev"] >= 0 and not p["excluded"]]
    rows.sort(key=lambda p: p["ev"], reverse=True)
    if not rows:
        print("\nNingún pick con EV >= 0: no se genera la tabla.")
        return

    col_labels = ["Partido", "Mercado", "Pick", "Cuota", "Prob.\nModelo",
                  "Prob.\nBookie", "EV", "Kelly\n25%", "Stake ($)", "Señal"]
    cell_text = [[
        p["partido"], p["mercado"], p["pick"], f"{p['cuota']:.2f}",
        f"{p['prob_model'] * 100:.1f}%", f"{p['prob_bookie'] * 100:.1f}%",
        f"{p['ev'] * 100:+.1f}%", f"{p['kelly'] * 100:.1f}%",
        f"${p['stake']:,.2f}", p["senal"],
    ] for p in rows]

    n_filas = len(rows) + 1  # datos + encabezado
    fig, ax = plt.subplots(figsize=(13.5, 0.40 * n_filas + 0.75), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.axis("off")
    ax.set_position((0.01, 0.03, 0.98, 0.97 - 0.55 / (0.40 * n_filas + 0.75)))
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     cellLoc="center", bbox=(0, 0, 1, 1),
                     colWidths=[0.15, 0.08, 0.15, 0.06, 0.08, 0.08, 0.07, 0.07, 0.09, 0.11])
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(1)
        if row == 0:
            cell.set_facecolor(SURFACE)
            cell.set_text_props(color=INK_2, fontweight="bold", fontsize=8.5)
        else:
            senal = rows[row - 1]["senal"]
            cell.set_facecolor(SIGNAL_STYLE[senal]["bg"] if col == 9 else SURFACE)
            cell.set_text_props(color=INK)

    ax.set_title("Picks con valor — Copa del Mundo 2026 (EV descendente)",
                 fontsize=13, fontweight="bold", loc="left", pad=6)
    os.makedirs(path.parent, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nTabla de picks guardada en {path}")


def print_summary(bankroll, picks):
    line = "═" * 50
    print()
    print(line)
    print("PICKS CON VALOR — Copa del Mundo 2026")
    print(f"Bankroll: ${bankroll:,.2f}")
    print(line)

    excluded = [p for p in picks if p["excluded"]]
    valid = [p for p in picks if not p["excluded"]]

    total_picks, total_stake = 0, 0.0
    for senal in ("VALOR ALTO", "VALOR MEDIO", "VALOR BAJO"):
        group = sorted((p for p in valid if p["senal"] == senal),
                       key=lambda p: p["ev"], reverse=True)
        if not group:
            continue
        suffix = " (proceder con cautela)" if senal == "VALOR BAJO" else ""
        print(f"\n{SIGNAL_STYLE[senal]['emoji']} {senal}{suffix}")
        for p in group:
            print(f"{p['partido']} | {p['mercado']} | Pick: {p['pick']}")
            print(f"Cuota: {p['cuota']:.2f} | EV: {p['ev'] * 100:+.1f}% | "
                  f"Stake sugerido: ${p['stake']:,.2f}")
            total_picks += 1
            total_stake += p["stake"]

    if excluded:
        print("\n🚫 ALTA INCERTIDUMBRE (excluidos: los modelos difieren >15 pp)")
        for p in excluded:
            print(f"{p['partido']} | {p['mercado']} | Pick: {p['pick']} | "
                  f"diferencia entre modelos: {p['uncertainty'] * 100:.1f} pp")

    no_value = sum(1 for p in valid if p["senal"] == "SIN VALOR")
    if no_value:
        print(f"\n❌ SIN VALOR: {no_value} mercados descartados (EV negativo)")

    print()
    print("─" * 50)
    pct = (total_stake / bankroll * 100) if bankroll else 0
    print(f"Total picks sugeridos: {total_picks}")
    print(f"Stake total comprometido: ${total_stake:,.2f} ({pct:.1f}% del bankroll)")
    print(line)
    print()
    print("⚠️  ADVERTENCIA: Estas predicciones son estimaciones estadísticas.")
    print("Ningún modelo garantiza resultados. Apuesta solo lo que puedas perder.")


def main():
    with open(BASE_DIR / "models" / "dixon_coles.pkl", "rb") as f:
        dc_model = pickle.load(f)
    with open(BASE_DIR / "models" / "xgboost_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)

    bankroll, picks = collect_inputs_and_analyze(dc_model, xgb_model)
    save_picks_table(picks, BASE_DIR / "output" / "betting_picks.png")
    print_summary(bankroll, picks)


if __name__ == "__main__":
    main()
