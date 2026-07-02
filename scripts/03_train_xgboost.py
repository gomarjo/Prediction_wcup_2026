# -*- coding: utf-8 -*-
"""
FASE 4 - Entrenamiento de los modelos XGBoost.

1. Clasificador de resultado en 90 min (0=local, 1=empate, 2=visita) con las
   features de la Fase 2. Validacion: ultimos 2 anios. Early stopping: 50.
2. Dos regresores de goles (home_score y away_score) que usan las mismas
   features mas las probabilidades predichas por el clasificador.

Guarda los tres modelos (con el historial de equipos para prediccion) en
models/xgboost_model.pkl.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier, XGBRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from match_features import FEATURE_COLS, build_feature_dataset
from xgb_model import REGRESSOR_COLS, XGBoostPredictor

XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)


def main():
    print("Construyendo features (Fase 2)...")
    features, tracker = build_feature_dataset()
    print(f"Partidos con features: {len(features):,}")

    # Ultimos 2 anios como validacion
    cutoff = features["date"].max() - np.timedelta64(730, "D")
    train = features[features["date"] < cutoff]
    valid = features[features["date"] >= cutoff]
    print(f"Entrenamiento: {len(train):,} | Validacion (desde {cutoff.date()}): {len(valid):,}")

    X_tr, y_tr = train[FEATURE_COLS], train["target"]
    X_va, y_va = valid[FEATURE_COLS], valid["target"]

    # --- Clasificador de resultado ---
    clf = XGBClassifier(
        **XGB_PARAMS, objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", early_stopping_rounds=50,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    probs_va = clf.predict_proba(X_va)
    acc = accuracy_score(y_va, probs_va.argmax(axis=1))
    ll = log_loss(y_va, probs_va)
    print(f"Clasificador -> accuracy validacion: {acc:.4f} | log-loss: {ll:.4f} "
          f"| mejor iteracion: {clf.best_iteration}")

    # --- Regresores de goles (features + probabilidades del clasificador) ---
    def with_clf_probs(X):
        probs = clf.predict_proba(X)
        X2 = X.copy()
        X2["clf_home_prob"] = probs[:, 0]
        X2["clf_draw_prob"] = probs[:, 1]
        X2["clf_away_prob"] = probs[:, 2]
        return X2[REGRESSOR_COLS]

    Xr_tr, Xr_va = with_clf_probs(X_tr), with_clf_probs(X_va)
    regressors = {}
    for name, col in (("home", "home_score"), ("away", "away_score")):
        reg = XGBRegressor(
            **XGB_PARAMS, objective="count:poisson",
            eval_metric="poisson-nloglik", early_stopping_rounds=50,
        )
        reg.fit(Xr_tr, train[col], eval_set=[(Xr_va, valid[col])], verbose=False)
        pred = reg.predict(Xr_va)
        mae = np.mean(np.abs(pred - valid[col]))
        print(f"Regresor {name}_score -> MAE validacion: {mae:.4f} "
              f"| mejor iteracion: {reg.best_iteration}")
        regressors[name] = reg

    predictor = XGBoostPredictor(clf, regressors["home"], regressors["away"], tracker)

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out = models_dir / "xgboost_model.pkl"
    with open(out, "wb") as f:
        pickle.dump(predictor, f)
    print(f"Modelos guardados en {out}")

    demo = predictor.predict("Spain", "Austria", neutral=True)
    print(f"Prueba Spain vs Austria (neutral): "
          f"G {demo['home_win_prob']:.3f} | E {demo['draw_prob']:.3f} | "
          f"P {demo['away_win_prob']:.3f} | xG {demo['home_xg']:.2f}-{demo['away_xg']:.2f}")


if __name__ == "__main__":
    main()
