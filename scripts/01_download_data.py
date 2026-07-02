# -*- coding: utf-8 -*-
"""
FASE 1 - Descarga de datos del repositorio martj42/international_results.
Descarga siempre la version mas reciente (sin cache) porque el repositorio
se actualiza cada noche.
"""
import os
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

FILES = {
    "results.csv": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    "goalscorers.csv": "https://raw.githubusercontent.com/martj42/international_results/master/goalscorers.csv",
    "shootouts.csv": "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv",
}

# Cabeceras para evitar cualquier cache intermedio
HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
}


def download_all():
    os.makedirs(DATA_DIR, exist_ok=True)
    for name, url in FILES.items():
        print(f"Descargando {name} ...")
        resp = requests.get(url, headers=HEADERS, timeout=120)
        resp.raise_for_status()
        dest = DATA_DIR / name
        dest.write_bytes(resp.content)
        print(f"  Guardado en {dest} ({len(resp.content):,} bytes)")


def summarize():
    results = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    print()
    print(f"Fecha del partido mas reciente en results.csv: {results['date'].max().date()}")
    print(f"Total de registros descargados (results.csv): {len(results):,}")
    for extra in ("goalscorers.csv", "shootouts.csv"):
        n = len(pd.read_csv(DATA_DIR / extra))
        print(f"Total de registros descargados ({extra}): {n:,}")


if __name__ == "__main__":
    download_all()
    summarize()
