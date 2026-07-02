# -*- coding: utf-8 -*-
"""
Clase del modelo Dixon-Coles (compartida entre entrenamiento y prediccion).

Los goles de cada equipo se modelan como Poisson con parametros de ataque
(alpha) y defensa (beta) por equipo, ventaja de local global (gamma) y una
correccion rho para marcadores bajos (0-0, 1-0, 0-1, 1-1).

    log(lambda_home) = alpha_home + beta_away + gamma * (1 si no es neutral)
    log(mu_away)     = alpha_away + beta_home

beta se interpreta como "debilidad defensiva": mas alto => concede mas goles.
"""
import numpy as np
from scipy.stats import poisson


class DixonColesModel:
    def __init__(self, attack, defence, gamma, rho, max_goals=10):
        self.attack = dict(attack)      # {equipo: alpha}
        self.defence = dict(defence)    # {equipo: beta}
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.max_goals = int(max_goals)

    def teams(self):
        return sorted(self.attack)

    def expected_goals(self, home_team, away_team, neutral=False):
        for t in (home_team, away_team):
            if t not in self.attack:
                raise KeyError(f"Equipo no encontrado en el modelo: {t!r}")
        home_adv = 0.0 if neutral else self.gamma
        lam = np.exp(self.attack[home_team] + self.defence[away_team] + home_adv)
        mu = np.exp(self.attack[away_team] + self.defence[home_team])
        return lam, mu

    def _tau_matrix(self, lam, mu, size):
        tau = np.ones((size, size))
        tau[0, 0] = 1.0 - lam * mu * self.rho
        tau[0, 1] = 1.0 + lam * self.rho
        tau[1, 0] = 1.0 + mu * self.rho
        tau[1, 1] = 1.0 - self.rho
        return np.clip(tau, 1e-10, None)

    def score_probability_matrix(self, home_team, away_team, neutral=False):
        """Matriz P[i, j] = prob(marcador i-j) para i, j en 0..max_goals."""
        lam, mu = self.expected_goals(home_team, away_team, neutral=neutral)
        goals = np.arange(self.max_goals + 1)
        p_home = poisson.pmf(goals, lam)
        p_away = poisson.pmf(goals, mu)
        matrix = np.outer(p_home, p_away) * self._tau_matrix(lam, mu, self.max_goals + 1)
        matrix /= matrix.sum()
        return matrix

    def predict(self, home_team, away_team, neutral=False):
        matrix = self.score_probability_matrix(home_team, away_team, neutral=neutral)
        idx_home, idx_away = np.indices(matrix.shape)
        home_win = matrix[idx_home > idx_away].sum()
        draw = np.trace(matrix)
        away_win = matrix[idx_home < idx_away].sum()
        goals = np.arange(matrix.shape[0])
        home_xg = float((matrix.sum(axis=1) * goals).sum())
        away_xg = float((matrix.sum(axis=0) * goals).sum())
        score_matrix = {
            (i, j): float(matrix[i, j]) for i in range(7) for j in range(7)
        }
        return {
            "home_win_prob": float(home_win),
            "draw_prob": float(draw),
            "away_win_prob": float(away_win),
            "home_xg": home_xg,
            "away_xg": away_xg,
            "score_matrix": score_matrix,
        }
