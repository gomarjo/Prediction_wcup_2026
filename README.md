# Predicción de partidos de fútbol internacional

Sistema de predicción de resultados de fútbol de selecciones (1X2, goles esperados,
marcadores exactos, goleadores y análisis de valor en apuestas) usando dos modelos
independientes: **Dixon-Coles** y **XGBoost**.

Los datos provienen del repositorio público
[martj42/international_results](https://github.com/martj42/international_results)
(resultados de partidos internacionales desde 1872, actualizado cada noche).

## Instalación

```bash
pip install -r requirements.txt
```

## Uso — pipeline completo

Los scripts se ejecutan en orden. Los datos, modelos y gráficos se generan
localmente (no van en el repo).

| Script | Qué hace |
|---|---|
| `scripts/01_download_data.py` | Descarga `results.csv`, `goalscorers.csv` y `shootouts.csv` a `data/` |
| `scripts/02_train_dixon_coles.py` | Entrena Dixon-Coles (Poisson con corrección de marcadores bajos, decaimiento temporal con semivida de 3 años) → `models/dixon_coles.pkl` |
| `scripts/03_train_xgboost.py` | Entrena clasificador G/E/P + 2 regresores de goles con features rodantes → `models/xgboost_model.pkl` |
| `scripts/04_predict_and_plot.py` | Predicciones y 3 gráficos por partido (ganador, xG, top-10 marcadores) → `output/partidoN/` |
| `scripts/05_betting_value.py` | Interactivo: compara probabilidades vs cuotas del mercado, EV y stake Kelly 25% → `output/betting_picks.png` |
| `scripts/06_goalscorers.py` | Probabilidad de anotar por jugador (anytime y 2+ goles) → `output/partidoN/partidoN_goleadores.png` |

Módulos compartidos: `dc_model.py` (clase Dixon-Coles), `xgb_model.py` (predictor
XGBoost), `match_features.py` (ingeniería de features point-in-time) y
`matches_config.py`.

## Cambiar la jornada

Edita **solo** la lista `MATCHES` en `scripts/matches_config.py`:

```python
MATCHES = [
    (1, "España", "Austria", "Spain", "Austria"),
    # (número, nombre local, nombre visitante, nombre en el dataset local, visitante)
]
```

Los nombres del dataset van en inglés (columnas `home_team`/`away_team` de
`results.csv`). Para partidos en sede neutral los scripts usan `neutral=True`.

## Modelos

- **Dixon-Coles**: goles como Poisson con parámetros de ataque/defensa por equipo,
  ventaja de local global (γ) y corrección ρ para marcadores bajos (0-0, 1-0, 0-1, 1-1).
  Máxima verosimilitud ponderada por recencia, optimizada con L-BFGS-B y gradiente analítico.
- **XGBoost**: clasificador multiclase (victoria/empate/derrota) + regresores Poisson de
  goles, con features rodantes calculadas solo con información previa a cada partido
  (fuerza de ataque/defensa, forma, head-to-head, ventaja de local, peso del torneo).
- **Goleadores**: reparto del xG consenso del equipo entre sus anotadores de los últimos
  2 años (decaimiento con semivida de 1 año, sin autogoles).

## Advertencia

Estas predicciones son estimaciones estadísticas; ningún modelo garantiza resultados.
El análisis de valor (script 05) es informativo. Si apuestas, hazlo con moderación y
solo lo que puedas permitirte perder.
