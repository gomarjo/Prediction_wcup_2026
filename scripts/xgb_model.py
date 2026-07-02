# -*- coding: utf-8 -*-
"""
Clase del predictor XGBoost (compartida entre entrenamiento y prediccion).

Envuelve el clasificador de resultado (G/E/P) y los dos regresores de goles,
junto con el historial de equipos (TeamHistoryTracker) actualizado hasta el
ultimo partido del dataset, para poder construir las features de un partido
futuro en el momento de la prediccion.
"""
import numpy as np
import pandas as pd

from match_features import FEATURE_COLS

REGRESSOR_COLS = FEATURE_COLS + ["clf_home_prob", "clf_draw_prob", "clf_away_prob"]


class XGBoostPredictor:
    def __init__(self, classifier, reg_home, reg_away, tracker,
                 tournament="FIFA World Cup"):
        self.classifier = classifier
        self.reg_home = reg_home
        self.reg_away = reg_away
        self.tracker = tracker
        self.tournament = tournament

    def teams(self):
        return sorted(self.tracker.scored)

    def _feature_row(self, home_team, away_team, neutral):
        for t in (home_team, away_team):
            if t not in self.tracker.scored:
                raise KeyError(f"Equipo no encontrado en el historial: {t!r}")
        feats = self.tracker.features_for_match(
            home_team, away_team, self.tournament, neutral)
        return pd.DataFrame([feats])[FEATURE_COLS]

    def predict(self, home_team, away_team, neutral=False):
        X = self._feature_row(home_team, away_team, neutral)
        probs = self.classifier.predict_proba(X)[0]  # [local, empate, visita]

        X_reg = X.copy()
        X_reg["clf_home_prob"] = probs[0]
        X_reg["clf_draw_prob"] = probs[1]
        X_reg["clf_away_prob"] = probs[2]
        X_reg = X_reg[REGRESSOR_COLS]
        home_xg = float(np.clip(self.reg_home.predict(X_reg)[0], 0.05, None))
        away_xg = float(np.clip(self.reg_away.predict(X_reg)[0], 0.05, None))

        return {
            "home_win_prob": float(probs[0]),
            "draw_prob": float(probs[1]),
            "away_win_prob": float(probs[2]),
            "home_xg": home_xg,
            "away_xg": away_xg,
        }
