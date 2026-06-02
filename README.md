# qoe-randomforest-squad

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20502846.svg)](https://doi.org/10.5281/zenodo.20502846)

Modelo de **Machine Learning basado en Random Forest** para la optimización de la
**Calidad de Experiencia (QoE)** en la red de una academia de música y
neurodesarrollo de alta densidad (Academia SQUAD, Bogotá, 2026).

Este repositorio acompaña a la tesis y reúne el **código del modelo**, el
**modelo entrenado** y su **evidencia**, para garantizar la transparencia y la
reproducibilidad del trabajo. Todo el software empleado es de código abierto.

## Qué hace

El modelo clasifica cada flujo de tráfico de la red en una de **tres categorías de
prioridad** —crítico, normal y de fondo— a partir del **comportamiento del flujo**
(no solo del puerto o el protocolo), lo que le permite operar incluso sobre tráfico
cifrado. Con esa clasificación, el sistema prioriza el tráfico sensible (por ejemplo,
la videoconferencia en tiempo real) para sostener la QoE en las horas de mayor
demanda.

## Contenido

| Archivo | Descripción |
|---|---|
| `agent_ml.py` | Agente de inferencia del Random Forest + entrenador (módulo `/train`) e integración con el cálculo del MOS estimado. |
| `etiquetado_trafico.py` | Etiquetado determinista del conjunto de entrenamiento por reglas de puerto/protocolo/servicio (crítico/normal/background). |
| `rf_model.joblib` | Modelo Random Forest entrenado (artefacto serializado). |
| `rf_model_meta.json` | Evidencia del modelo: hiperparámetros, OOB, importancia de variables, mapeos de clase. |
| `MODEL_CARD.md` | Ficha del modelo (model card). |

## Resumen del modelo

- Algoritmo: `RandomForestClassifier` (scikit-learn), 100 árboles, `random_state=42`.
- Entradas: **10 características** de comportamiento del flujo.
- Salidas: **3 clases** de prioridad (crítico, normal, de fondo).
- Conjunto de entrenamiento: **2 295 flujos reales**, balanceado (**765 por clase**).
- **Error Out-of-Bag (OOB): 4.01 %** (`oob_score` = 0.9599).

Importancia de las variables (las más relevantes): `rtp_payload_type` (0.238),
`bytes_per_second` (0.220), `inter_arrival_time` (0.180), `packet_size` (0.129).

> Nota de no circularidad: de los campos usados para etiquetar (puerto, protocolo,
> IP destino, `rtp_payload_type`), el modelo solo recibe como característica
> `rtp_payload_type`; aprende del comportamiento del flujo (9 de 10 características),
> por lo que generaliza a tráfico donde el puerto no basta.

## Requisitos

- Python 3.10+
- `scikit-learn`, `pandas`, `numpy`, `joblib`

```bash
pip install scikit-learn pandas numpy joblib
```

## Uso

Etiquetado del conjunto de entrenamiento:

```bash
python etiquetado_trafico.py flujos_capturados.csv flujos_etiquetados.csv
```

Reproducibilidad del modelo: re-entrenar con `n_estimators=100`, `random_state=42`,
`oob_score=True` sobre las 10 características reproduce un OOB de 0.0401.

## Privacidad

El código no incluye datos de tráfico reales ni direcciones IP de la institución.
Las IP de los servidores institucionales se reemplazaron por direcciones de
ejemplo del rango de documentación (RFC 5737, `192.0.2.0/24`).

## Licencia

MIT. Ver [LICENSE](LICENSE).

## Cómo citar

DOI (concepto, todas las versiones): **10.5281/zenodo.20502846** ·
https://doi.org/10.5281/zenodo.20502846

> Bernardo Livaque, R. A., y Rojas Sánchez, J. C. (2026). *qoe-randomforest-squad: modelo Random Forest para la optimización de la Calidad de Experiencia (QoE) en una academia de alta densidad* (Versión 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.20502846

Ver también [CITATION.cff](CITATION.cff). Este repositorio es material complementario
de la tesis sobre optimización de la QoE mediante Random Forest (Academia SQUAD,
Bogotá, 2026).
