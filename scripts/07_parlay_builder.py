# -*- coding: utf-8 -*-
"""
Constructor de parleys (bet builder) — evalua combinaciones de patas del
MISMO partido calculando la probabilidad conjunta REAL sobre la matriz de
marcadores (Dixon-Coles y XGBoost), respetando las correlaciones entre patas
(multiplicar probabilidades sueltas sobreestima el combo).

Patas soportadas:
    1. Resultado (1 / X / 2 / 1X / 12 / X2)
    2. Total de goles (mas/menos de N.5)
    3. Ambos equipos anotan (si/no)
    4. Marcador exacto (ej. 2-0)
    5. Goleador anota en cualquier momento (varios permitidos)
    6. Goles de un equipo (mas/menos de N.5)

Si ingresas la cuota del bookie calcula EV, senal y stake Kelly 25%
(cap 5% del bankroll). Interactivo, en bucle hasta que decidas salir.
"""
import pickle
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.stats import poisson

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from matches_config import MATCHES  # noqa: E402

MAX_GOALS = 10
UNCERTAINTY_LIMIT = 0.15
KELLY_FRACTION = 0.25
STAKE_CAP = 0.05


# ---------------------------------------------------------------- entrada
def ask(prompt, valid=None, allow_blank=False):
    while True:
        raw = input(prompt).strip()
        if not raw and allow_blank:
            return None
        value = raw.lower()
        if valid is None and raw:
            return raw
        if valid is not None and value in valid:
            return value
        print(f"  Opción inválida. Opciones: {', '.join(sorted(valid))}" if valid
              else "  Valor requerido.")


def ask_number(prompt, minimum=0.0, allow_blank=False):
    while True:
        raw = input(prompt).strip().replace(",", ".").replace("$", "")
        if not raw and allow_blank:
            return None
        try:
            value = float(raw)
            if value >= minimum:
                return value
        except ValueError:
            pass
        print(f"  Número inválido (mínimo {minimum}).")


# ---------------------------------------------------------------- modelo
def load_models_and_shares():
    with open(BASE_DIR / "models" / "dixon_coles.pkl", "rb") as f:
        dc = pickle.load(f)
    with open(BASE_DIR / "models" / "xgboost_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    # Shares de goleadores (mismo modelo que el script 06)
    spec_path = Path(__file__).resolve().parent / "06_goalscorers.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("g06", spec_path)
    g06 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g06)
    goals_df = g06.load_recent_goals()
    return dc, xgb, g06, goals_df


def matrices_for(dc, xgb, home_ds, away_ds):
    m_dc = dc.score_probability_matrix(home_ds, away_ds, neutral=True)
    pred = xgb.predict(home_ds, away_ds, neutral=True)
    g = np.arange(MAX_GOALS + 1)
    m_xgb = np.outer(poisson.pmf(g, pred["home_xg"]), poisson.pmf(g, pred["away_xg"]))
    m_xgb /= m_xgb.sum()
    return m_dc, m_xgb


def prob_all_scorers(shares, n_goals):
    """P(todos los jugadores con esos shares anotan | el equipo mete n_goals),
    por inclusion-exclusion sobre el reparto multinomial de goles."""
    total = 0.0
    for r in range(len(shares) + 1):
        for subset in combinations(shares, r):
            total += (-1) ** r * max(0.0, 1.0 - sum(subset)) ** n_goals
    return total


def joint_probability(matrix, mask, scorers_home, scorers_away):
    total = 0.0
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            if not mask[i, j]:
                continue
            factor = matrix[i, j]
            if scorers_home:
                factor *= prob_all_scorers(scorers_home, i)
            if scorers_away:
                factor *= prob_all_scorers(scorers_away, j)
            total += factor
    return total


# ---------------------------------------------------------------- patas
def build_legs(home_es, away_es, shares_home, shares_away):
    """Bucle interactivo: retorna (mask, scorers_home, scorers_away, etiquetas)."""
    ih, ij = np.indices((MAX_GOALS + 1, MAX_GOALS + 1))
    mask = np.ones_like(ih, dtype=bool)
    scorers_home, scorers_away, labels = [], [], []

    menu = f"""
  Agregar pata:
    1) Resultado (1 / X / 2 / 1X / 12 / X2)
    2) Total de goles (más/menos de N.5)
    3) Ambos equipos anotan (sí/no)
    4) Marcador exacto (ej. 2-0)
    5) Goleador anota ({home_es} o {away_es})
    6) Goles de un equipo (más/menos de N.5)
    0) Listo, calcular"""

    while True:
        print(menu)
        op = ask("  Opción: ", valid={"0", "1", "2", "3", "4", "5", "6"})
        if op == "0":
            if not labels:
                print("  Agrega al menos una pata.")
                continue
            return mask, scorers_home, scorers_away, labels

        if op == "1":
            r = ask("  Resultado (1/X/2/1X/12/X2): ",
                    valid={"1", "x", "2", "1x", "12", "x2"})
            conds = {"1": ih > ij, "x": ih == ij, "2": ih < ij,
                     "1x": ih >= ij, "12": ih != ij, "x2": ih <= ij}
            names = {"1": f"Gana {home_es}", "x": "Empate", "2": f"Gana {away_es}",
                     "1x": f"{home_es} o Empate", "12": "Cualquiera gana",
                     "x2": f"Empate o {away_es}"}
            mask &= conds[r]
            labels.append(names[r])

        elif op == "2":
            d = ask("  ¿Más o menos? (mas/menos): ", valid={"mas", "menos"})
            linea = ask_number("  Línea (ej. 2.5): ", minimum=0.5)
            mask &= (ih + ij > linea) if d == "mas" else (ih + ij < linea)
            labels.append(f"{'Más' if d == 'mas' else 'Menos'} de {linea} goles")

        elif op == "3":
            r = ask("  ¿Ambos anotan? (si/no): ", valid={"si", "no"})
            mask &= ((ih > 0) & (ij > 0)) if r == "si" else ((ih == 0) | (ij == 0))
            labels.append(f"Ambos anotan: {'Sí' if r == 'si' else 'No'}")

        elif op == "4":
            while True:
                raw = ask(f"  Marcador exacto local-visita (ej. 2-0): ")
                try:
                    gi, gj = (int(x) for x in raw.replace(" ", "").split("-"))
                    break
                except ValueError:
                    print("  Formato inválido. Ejemplo: 2-0")
            mask &= (ih == gi) & (ij == gj)
            labels.append(f"Exacto {home_es} {gi} - {gj} {away_es}")

        elif op == "5":
            side = ask(f"  ¿Equipo? (1={home_es}, 2={away_es}): ", valid={"1", "2"})
            shares = shares_home if side == "1" else shares_away
            team_es = home_es if side == "1" else away_es
            print(f"  Goleadores de {team_es}:")
            top = shares.head(10)
            for k, row in enumerate(top.itertuples(index=False), start=1):
                print(f"    {k}) {row.scorer}  (share {row.share * 100:.1f}%)")
            while True:
                n = ask_number("  Número del jugador: ", minimum=1)
                if n and int(n) <= len(top):
                    break
                print("  Fuera de rango.")
            row = top.iloc[int(n) - 1]
            (scorers_home if side == "1" else scorers_away).append(float(row["share"]))
            labels.append(f"{row['scorer']} anota ({team_es})")

        elif op == "6":
            side = ask(f"  ¿Equipo? (1={home_es}, 2={away_es}): ", valid={"1", "2"})
            d = ask("  ¿Más o menos? (mas/menos): ", valid={"mas", "menos"})
            linea = ask_number("  Línea (ej. 1.5): ", minimum=0.5)
            goles = ih if side == "1" else ij
            team_es = home_es if side == "1" else away_es
            mask &= (goles > linea) if d == "mas" else (goles < linea)
            labels.append(f"{team_es}: {'más' if d == 'mas' else 'menos'} de {linea} goles")


# ---------------------------------------------------------------- principal
def analyze_one(dc, xgb, g06, goals_df):
    print("\nPartidos disponibles:")
    for num, h, a, _, _ in MATCHES:
        print(f"  {num}) {h} vs {a}")
    valid = {str(m[0]) for m in MATCHES}
    num = ask("Elige el partido: ", valid=valid)
    _, home_es, away_es, home_ds, away_ds = next(m for m in MATCHES if str(m[0]) == num)

    m_dc, m_xgb = matrices_for(dc, xgb, home_ds, away_ds)
    shares_home = g06.team_scorer_shares(goals_df, home_ds)
    shares_away = g06.team_scorer_shares(goals_df, away_ds)

    mask, sc_h, sc_a, labels = build_legs(home_es, away_es, shares_home, shares_away)

    p_dc = joint_probability(m_dc, mask, sc_h, sc_a)
    p_xgb = joint_probability(m_xgb, mask, sc_h, sc_a)
    consensus = (p_dc + p_xgb) / 2
    diff = abs(p_dc - p_xgb)

    print("\n" + "─" * 52)
    print(f"Parley: {home_es} vs {away_es}")
    for lab in labels:
        print(f"  • {lab}")
    print("─" * 52)
    print(f"  Dixon-Coles: {p_dc * 100:.1f}%  |  XGBoost: {p_xgb * 100:.1f}%")
    print(f"  CONSENSO: {consensus * 100:.1f}%  →  cuota justa: "
          f"{1 / consensus:.2f}" if consensus > 0 else "  CONSENSO: 0% (imposible)")
    if consensus > 0:
        print(f"  Pierde aprox. {round(1 / consensus) - 1} de cada "
              f"{round(1 / consensus)} veces")
    if diff > UNCERTAINTY_LIMIT:
        print(f"  🚫 ALTA INCERTIDUMBRE: los modelos difieren {diff * 100:.1f} pp")

    if consensus <= 0:
        print("  ❌ Combo imposible: las patas se contradicen entre sí.")
        return

    cuota = ask_number("\n¿Cuota que te ofrece el bookie? (Enter para omitir): ",
                       minimum=1.01, allow_blank=True)
    if cuota:
        ev = consensus * cuota - 1
        senal = ("✅ VALOR ALTO" if ev >= 0.05 else "⚠️ VALOR MEDIO" if ev >= 0.02
                 else "🔍 VALOR BAJO" if ev >= 0 else "❌ SIN VALOR")
        print(f"  EV: {ev * 100:+.1f}%  →  {senal}")
        if ev > 0:
            bankroll = ask_number("¿Bankroll? $ (Enter para omitir): ",
                                  minimum=1, allow_blank=True)
            if bankroll:
                kelly = (consensus * cuota - 1) / (cuota - 1) * KELLY_FRACTION
                stake = bankroll * min(max(kelly, 0), STAKE_CAP)
                print(f"  Stake sugerido (Kelly 25%, cap 5%): ${stake:,.2f}")
        else:
            print(f"  Para tener valor necesitas cuota > {1 / consensus:.2f}")


def main():
    print("═" * 52)
    print("CONSTRUCTOR DE PARLEYS — Copa del Mundo 2026")
    print("Probabilidad conjunta real (correlaciones incluidas)")
    print("═" * 52)
    dc, xgb, g06, goals_df = load_models_and_shares()
    try:
        while True:
            analyze_one(dc, xgb, g06, goals_df)
            again = ask("\n¿Analizar otro parley? (si/no): ", valid={"si", "no"})
            if again == "no":
                break
    except EOFError:
        pass
    print("\n⚠️  Estimaciones estadísticas; ningún modelo garantiza resultados.")


if __name__ == "__main__":
    main()
