# Model Card — qoe-randomforest-squad

## Descripción
Modelo de clasificación de tráfico de red basado en **Random Forest**, entrenado
con tráfico real capturado y etiquetado de la red de la Academia SQUAD durante la
fase de diseño y calibración (previa al periodo de observación). Su salida (clase
de prioridad) alimenta la política de priorización que optimiza la QoE.

## Detalles técnicos
- **Algoritmo:** RandomForestClassifier (scikit-learn)
- **Árboles (`n_estimators`):** 100
- **`random_state`:** 42
- **Características de entrada:** 10
- **Clases de salida:** 3 (0 = crítico, 1 = normal, 2 = background)
- **Muestras de entrenamiento:** 2 295 (balanceado: 765 por clase)
- **Error Out-of-Bag (OOB):** 0.0401 (≈ 4.01 %) · `oob_score` = 0.9599

## Características (features)
`packet_size`, `inter_arrival_time`, `jitter`, `packet_loss_rate`,
`bandwidth_utilization`, `rtp_payload_type`, `burst_duration`, `flow_duration`,
`packets_per_second`, `bytes_per_second`.

### Importancia de las características
| Característica | Importancia |
|---|---|
| rtp_payload_type | 0.2382 |
| bytes_per_second | 0.2201 |
| inter_arrival_time | 0.1799 |
| packet_size | 0.1294 |
| flow_duration | 0.0577 |
| burst_duration | 0.0572 |
| jitter | 0.0516 |
| packets_per_second | 0.0310 |
| bandwidth_utilization | 0.0280 |
| packet_loss_rate | 0.0070 |

## Mapeo de clases a servicios
- **Crítico:** VoIP, Videoconferencia
- **Normal:** Web educativa
- **Background:** Streaming, Transferencia de archivos, Juegos, Otros

## Traducción a prioridad de red
- Crítico → DSCP EF → cola HTB 1:10
- Normal → DSCP AF31 → cola HTB 1:20
- Background → cola HTB 1:30 (el marcado DSCP se refina por subtipo: AF22 / AF11 / BE
  en una segunda capa, lo que explica las cinco marcas DSCP que registra la telemetría
  aunque el modelo decida tres clases).

## Reproducibilidad
Re-entrenar con `n_estimators=100`, `random_state=42`, `oob_score=True` sobre las
10 características reproduce un OOB de 0.0401.

## Limitaciones
- El MOS asociado se reporta siempre como **MOS estimado** (función paramétrica
  inspirada en ITU-T P.1203 simplificada); no proviene de encuestas de percepción.
- El conjunto de entrenamiento está balanceado por clase; la distribución real de
  producción se observa por separado en la telemetría.
- La calibración del umbral en producción y la validación del MOS estimado contra
  MOS subjetivo quedan como trabajo futuro.

## Licencia
MIT.
