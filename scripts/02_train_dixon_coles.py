# -*- coding: utf-8 -*-
"""
FASE 3 - Entrenamiento del modelo Dixon-Coles.

Estima por maxima verosimilitud (ponderada por decaimiento temporal con
semivida de 3 anios) los parametros alpha/beta por equipo, gamma (ventaja
de local) y rho (correccion de marcadores bajos), con L-BFGS-B y gradiente
analitico. Guarda el modelo en models/dixon_coles.pkl.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dc_model import DixonColesModel

MIN_DATE = "2010-01-01"
HALF_LIFE_DAYS = 1095  # 3 anios
DECAY_LAMBDA = np.log(2) / HALF_LIFE_DAYS


def load_matches():
    df = pd.read_csv(BASE_DIR / "data" / "results.csv", parse_dates=["date"])
    df = df[df["date"] >= MIN_DATE].dropna(subset=["home_score", "away_score"])
    # Solo partidos ya jugados (el repositorio puede listar partidos futuros)
    df = df[df["date"] <= pd.Timestamp.today()].reset_index(drop=True)
    return df


def negative_log_likelihood(params, n_teams, h_idx, a_idx, x, y, home_ind, w,
                            return_grad=True):
    attack = params[:n_teams]
    defence = params[n_teams:2 * n_teams]
    gamma, rho = params[-2], params[-1]

    loglam = attack[h_idx] + defence[a_idx] + gamma * home_ind
    logmu = attack[a_idx] + defence[h_idx]
    lam, mu = np.exp(loglam), np.exp(logmu)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    tau = np.ones_like(lam)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    tau = np.clip(tau, 1e-10, None)

    ll = np.sum(w * (np.log(tau) + x * loglam - lam + y * logmu - mu))
    # Penalizacion suave para fijar el grado de libertad redundante
    # (sumar c a todos los alpha y restarlo de todos los beta no cambia nada)
    penalty = 0.1 * attack.sum() ** 2
    nll = -ll + penalty
    if not return_grad:
        return nll

    # Gradiente analitico
    d_loglam = x - lam
    d_logmu = y - mu
    d_rho = np.zeros_like(lam)

    d_loglam[m00] += -lam[m00] * mu[m00] * rho / tau[m00]
    d_logmu[m00] += -lam[m00] * mu[m00] * rho / tau[m00]
    d_rho[m00] = -lam[m00] * mu[m00] / tau[m00]

    d_loglam[m01] += lam[m01] * rho / tau[m01]
    d_rho[m01] = lam[m01] / tau[m01]

    d_logmu[m10] += mu[m10] * rho / tau[m10]
    d_rho[m10] = mu[m10] / tau[m10]

    d_rho[m11] = -1.0 / tau[m11]

    A = w * d_loglam
    B = w * d_logmu
    grad_attack = np.bincount(h_idx, A, n_teams) + np.bincount(a_idx, B, n_teams)
    grad_defence = np.bincount(a_idx, A, n_teams) + np.bincount(h_idx, B, n_teams)
    grad_gamma = np.sum(A * home_ind)
    grad_rho = np.sum(w * d_rho)

    grad = -np.concatenate([grad_attack, grad_defence, [grad_gamma], [grad_rho]])
    grad[:n_teams] += 0.2 * attack.sum()
    return nll, grad


def fit(df):
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    h_idx = df["home_team"].map(team_idx).to_numpy()
    a_idx = df["away_team"].map(team_idx).to_numpy()
    x = df["home_score"].to_numpy(dtype=float)
    y = df["away_score"].to_numpy(dtype=float)
    home_ind = (~df["neutral"].astype(bool)).to_numpy(dtype=float)

    days_since = (df["date"].max() - df["date"]).dt.days.to_numpy(dtype=float)
    w = np.exp(-DECAY_LAMBDA * days_since)

    # Inicializacion razonable: beta absorbe la media global de goles
    mean_goals = (x.sum() + y.sum()) / (2 * len(df))
    x0 = np.concatenate([
        np.zeros(n),                        # alpha
        np.full(n, np.log(mean_goals)),     # beta
        [0.25, -0.05],                      # gamma, rho
    ])
    bounds = [(None, None)] * (2 * n) + [(None, None), (-0.2, 0.2)]

    result = minimize(
        negative_log_likelihood, x0, method="L-BFGS-B", jac=True, bounds=bounds,
        args=(n, h_idx, a_idx, x, y, home_ind, w),
        options={"maxiter": 3000, "maxfun": 30000},
    )
    print(f"Optimizacion terminada: {result.message}")
    print(f"  NLL final: {result.fun:.2f} | iteraciones: {result.nit}")

    attack = dict(zip(teams, result.x[:n]))
    defence = dict(zip(teams, result.x[n:2 * n]))
    gamma, rho = result.x[-2], result.x[-1]
    print(f"  gamma (ventaja local) = {gamma:.4f} | rho = {rho:.4f}")
    return DixonColesModel(attack, defence, gamma, rho)


def main():
    df = load_matches()
    print(f"Partidos de entrenamiento (desde {MIN_DATE}): {len(df):,}")
    model = fit(df)

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out = models_dir / "dixon_coles.pkl"
    with open(out, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en {out}")

    # Prueba rapida
    demo = model.predict("Spain", "Austria", neutral=True)
    print(f"Prueba Spain vs Austria (neutral): "
          f"G {demo['home_win_prob']:.3f} | E {demo['draw_prob']:.3f} | "
          f"P {demo['away_win_prob']:.3f} | xG {demo['home_xg']:.2f}-{demo['away_xg']:.2f}")


if __name__ == "__main__":
    main()
