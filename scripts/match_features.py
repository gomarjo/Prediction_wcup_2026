# -*- coding: utf-8 -*-
"""
FASE 2 - Ingenieria de features por partido (compartida entre entrenamiento
y prediccion). Todas las features se calculan solo con partidos ANTERIORES
al partido en cuestion (point-in-time, sin fuga de informacion).

Features por equipo:
    attack_strength   -> promedio de goles anotados en los ultimos 30 partidos
    defense_weakness  -> promedio de goles recibidos en los ultimos 30 partidos
    form_points       -> puntos en los ultimos 10 partidos / 30 (0-1)
    h2h_win_rate      -> tasa de victorias en los ultimos 10 enfrentamientos directos
    home_advantage    -> diferencia promedio de goles cuando juega de local
                         (solo partidos con neutral=FALSE)
    tournament_weight -> peso del torneo
"""
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MIN_DATE = "2010-01-01"

FEATURE_COLS = [
    "home_attack", "home_defense", "home_form", "home_h2h", "home_homeadv",
    "away_attack", "away_defense", "away_form", "away_h2h", "away_homeadv",
    "tournament_weight", "neutral_flag",
]


def tournament_weight(name):
    name = str(name)
    if name == "FIFA World Cup":
        return 1.0
    if name in ("Copa América", "UEFA Euro", "African Cup of Nations"):
        return 0.9
    if "qualification" in name.lower():
        return 0.8
    if name == "Friendly":
        return 0.5
    return 0.7  # otros torneos oficiales (Nations League, Gold Cup, etc.)


def load_matches():
    df = pd.read_csv(BASE_DIR / "data" / "results.csv", parse_dates=["date"])
    df = df[df["date"] >= MIN_DATE].dropna(subset=["home_score", "away_score"])
    df = df[df["date"] <= pd.Timestamp.today()]
    return df.sort_values("date").reset_index(drop=True)


def _deque30():
    return deque(maxlen=30)


def _deque10():
    return deque(maxlen=10)


class TeamHistoryTracker:
    """Mantiene el historial rodante de cada equipo y de cada par (h2h)."""

    def __init__(self):
        self.scored = defaultdict(_deque30)
        self.conceded = defaultdict(_deque30)
        self.points = defaultdict(_deque10)
        self.home_diffs = defaultdict(list)   # diffs de gol como local real
        self.h2h = defaultdict(_deque10)      # (par) -> ganador o 'draw'

    @staticmethod
    def _pair(a, b):
        return tuple(sorted((a, b)))

    def team_features(self, team):
        scored = self.scored[team]
        attack = float(np.mean(scored)) if len(scored) >= 5 else np.nan
        conceded = self.conceded[team]
        defense = float(np.mean(conceded)) if len(conceded) >= 5 else np.nan
        pts = self.points[team]
        form = sum(pts) / 30.0 if len(pts) == 10 else (sum(pts) / (3.0 * len(pts)) if pts else np.nan)
        diffs = self.home_diffs[team]
        home_adv = float(np.mean(diffs[-30:])) if diffs else 0.0
        return attack, defense, form, home_adv

    def h2h_win_rate(self, team, opponent):
        meetings = self.h2h[self._pair(team, opponent)]
        if not meetings:
            return 0.5  # sin historial: neutro
        wins = sum(1 for winner in meetings if winner == team)
        return wins / len(meetings)

    def features_for_match(self, home_team, away_team, tournament, neutral):
        h_att, h_def, h_form, h_homeadv = self.team_features(home_team)
        a_att, a_def, a_form, a_homeadv = self.team_features(away_team)
        return {
            "home_attack": h_att, "home_defense": h_def, "home_form": h_form,
            "home_h2h": self.h2h_win_rate(home_team, away_team),
            "home_homeadv": h_homeadv,
            "away_attack": a_att, "away_defense": a_def, "away_form": a_form,
            "away_h2h": self.h2h_win_rate(away_team, home_team),
            "away_homeadv": a_homeadv,
            "tournament_weight": tournament_weight(tournament),
            "neutral_flag": 1.0 if neutral else 0.0,
        }

    def update(self, row):
        h, a = row.home_team, row.away_team
        hs, as_ = row.home_score, row.away_score
        self.scored[h].append(hs)
        self.conceded[h].append(as_)
        self.scored[a].append(as_)
        self.conceded[a].append(hs)
        if hs > as_:
            self.points[h].append(3); self.points[a].append(0)
            self.h2h[self._pair(h, a)].append(h)
        elif hs < as_:
            self.points[h].append(0); self.points[a].append(3)
            self.h2h[self._pair(h, a)].append(a)
        else:
            self.points[h].append(1); self.points[a].append(1)
            self.h2h[self._pair(h, a)].append("draw")
        if not row.neutral:
            self.home_diffs[h].append(hs - as_)


def build_feature_dataset(df=None):
    """Retorna (features_df, tracker) donde features_df tiene una fila por
    partido con FEATURE_COLS + target + goles, y tracker queda actualizado
    con TODO el historial (util para predecir partidos futuros)."""
    if df is None:
        df = load_matches()
    tracker = TeamHistoryTracker()
    rows = []
    for row in df.itertuples(index=False):
        feats = tracker.features_for_match(
            row.home_team, row.away_team, row.tournament, row.neutral)
        feats["date"] = row.date
        feats["home_score"] = row.home_score
        feats["away_score"] = row.away_score
        if row.home_score > row.away_score:
            feats["target"] = 0
        elif row.home_score == row.away_score:
            feats["target"] = 1
        else:
            feats["target"] = 2
        rows.append(feats)
        tracker.update(row)
    return pd.DataFrame(rows), tracker
