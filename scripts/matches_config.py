# -*- coding: utf-8 -*-
"""
Partidos de la jornada a predecir (compartido por los scripts 04, 05 y 06).
Editar SOLO aqui para cambiar la jornada.

Formato: (numero, nombre_local_es, nombre_visita_es, local_dataset, visita_dataset)
El nombre *_dataset debe coincidir con results.csv (en ingles).
Todos se predicen con neutral=True (Copa del Mundo 2026, sede neutral).
"""

MATCHES = [
    (1, "España", "Austria", "Spain", "Austria"),
    (2, "Portugal", "Croacia", "Portugal", "Croatia"),
    (3, "Suiza", "Argelia", "Switzerland", "Algeria"),
]
