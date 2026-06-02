"""
================================================================================
AGENTE ML - RANDOM FOREST PARA CLASIFICACION DE TRAFICO QoE
Tesis: Optimizacion de la QoE mediante Random Forest - Academia SQUAD, Bogota 2026

Este programa es el nucleo del sistema de gestion de trafico. Recibe las
caracteristicas de cada flujo que circula por la red, las clasifica en una de
tres categorias de prioridad (critico, normal o background) mediante un modelo
Random Forest, y aplica la accion de priorizacion correspondiente sobre el
router de Linux mediante colas de Control de Trafico (tc/HTB).

--------------------------------------------------------------------------------
NOTA METODOLOGICA IMPORTANTE (modelo cientifico vs arranque en frio)
--------------------------------------------------------------------------------
El modelo Random Forest de esta investigacion se entrena con TRAFICO REAL
capturado y etiquetado en la red de la academia, a traves del endpoint /train,
durante la fase de diseno y calibracion (previa al segmento inicial de
observacion). El error Out-of-Bag (OOB) que se reporta en la tesis se calcula
sobre ese conjunto de entrenamiento real, NO sobre los datos de los meses de
observacion.

La funcion de arranque en frio (cold_start_model) genera un modelo provisional
con datos sinteticos UNICAMENTE para que el servicio web no falle si arranca sin
un modelo entrenado en disco. Ese modelo provisional NO es el modelo cientifico
del estudio y queda explicitamente marcado como tal en sus metadatos
('origen': 'cold_start_sintetico'). En produccion siempre debe cargarse el
modelo entrenado con datos reales.
================================================================================
"""

# =============================================================================
# IMPORTS - Librerias externas que el programa necesita usar
# =============================================================================

# Modulos estandar de Python:
# - os: para leer/verificar archivos del sistema (ej: ver si el modelo existe)
# - time: para medir cuanto tarda el modelo en predecir
# - logging: para escribir mensajes de registro (debugging)
# - subprocess: para ejecutar comandos del sistema operativo (tc de Linux)
# - json: para leer/escribir los metadatos del modelo (origen, fecha, OOB)
import os, time, logging, subprocess, json

# numpy: manejo de arreglos numericos (como vectores y matrices)
import numpy as np

# pandas: manejo de tablas de datos tipo Excel (CSV, DataFrames)
import pandas as pd

# joblib: libreria para guardar/cargar modelos de ML en disco
import joblib

# Flask: framework web para crear el API REST que expone endpoints HTTP
from flask import Flask, request, jsonify

# prometheus_client: libreria para exponer metricas a Prometheus (Grafana las lee)
# - Counter: contador que solo sube (ej: total inferencias hechas)
# - Gauge: valor que puede subir o bajar (ej: latencia actual)
# - Histogram: distribucion de valores (ej: tiempos de inferencia)
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# El corazon del programa: el clasificador Random Forest de scikit-learn
from sklearn.ensemble import RandomForestClassifier

# deque: lista optimizada con tamano maximo (cuando se llena, elimina los viejos)
from collections import deque


# =============================================================================
# CONFIGURACION DE LOGGING (mensajes en consola)
# =============================================================================
# Configura el formato de los mensajes: [fecha] [nivel] [mensaje]
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Se crea la aplicacion Flask que manejara las peticiones HTTP
app = Flask(__name__)


# =============================================================================
# METRICAS DE PROMETHEUS
# Cada metrica es una variable que Prometheus lee periodicamente
# para graficar en Grafana.
# =============================================================================

# Counter: cuenta total de inferencias que ha hecho el modelo desde que arranco
inference_counter    = Counter('rf_inferences_total', 'Total inferencias')

# Histogram: distribucion del tiempo que tarda el modelo en predecir
# Los buckets son los rangos: 1ms, 2ms, 5ms, 10ms, 20ms, 50ms, 100ms, 200ms
inference_latency    = Histogram('rf_inference_duration_ms', 'Latencia inferencia E2E (ms)', buckets=[1,2,5,10,20,50,100,200])

# Counter con etiqueta 'traffic_class': cuantos flujos clasifico como
# critico, normal o background (separado por clase)
packets_classified   = Counter('packets_classified_total', 'Flujos clasificados', ['traffic_class'])

# Gauges para las 5 metricas de Calidad de Experiencia (QoE)
# Estos valores se actualizan cada vez que el agente recibe datos
qoe_latency_gauge    = Gauge('qoe_latency_ms', 'Latencia de red (ms)')
qoe_jitter_gauge     = Gauge('qoe_jitter_ms', 'Jitter (ms)')
qoe_loss_gauge       = Gauge('qoe_packet_loss_pct', 'Perdida de paquetes (%)')
qoe_buffering_gauge  = Gauge('qoe_buffering_events', 'Eventos buffering/min')
qoe_resolution_gauge = Gauge('qoe_resolution_optimal_pct', '% tiempo resolucion optima')

# MOS estimado: la metrica principal, resultado del modelo parametrico propio
qoe_mos_gauge        = Gauge('qoe_mos_estimated', 'MOS estimado (1-5)')

# OOB Error: indicador de precision del modelo Random Forest
model_oob_gauge      = Gauge('rf_oob_error', 'OOB Error del modelo RF')

# Counter de reglas TC (Traffic Control de Linux) aplicadas por clase
tc_rules_applied     = Counter('tc_rules_applied_total', 'Reglas tc aplicadas', ['traffic_class'])


# =============================================================================
# CONFIGURACION DEL MODELO
# =============================================================================

# Ruta donde se guarda el modelo entrenado (para no perderlo al reiniciar)
MODEL_PATH = '/app/data/rf_model.joblib'

# Ruta donde se guardan los metadatos del modelo: origen (real o sintetico),
# fecha de entrenamiento, numero de muestras y OOB. Sirve como evidencia y para
# distinguir el modelo cientifico del arranque en frio.
META_PATH  = '/app/data/rf_model_meta.json'

# Numero de arboles del bosque. Se usa el MISMO valor (100) tanto en el
# arranque en frio como en el reentrenamiento con datos reales, para que el
# modelo desplegado y el reportado en la tesis sean identicos en este parametro.
N_ESTIMATORS = 100

# Variables globales: model guarda el clasificador entrenado; model_trained dice si esta listo
model = None
model_trained = False
# Metadatos del modelo actualmente cargado (origen, OOB, fecha, n_muestras)
model_meta = {}

# Diccionario que traduce el numero de clase a nombre humano
# El modelo predice 0, 1 o 2, y nosotros lo mostramos como "critico", "normal" o "background"
TRAFFIC_CLASSES = {0: 'critico', 1: 'normal', 2: 'background'}

# -----------------------------------------------------------------------------
# MAPEO DE LAS 3 CLASES DE PRIORIDAD A LOS SERVICIOS DE LA RED
# El modelo decide entre TRES clases de prioridad. Cada clase agrupa uno o varios
# de los servicios que conviven en la red de la academia, segun su sensibilidad a
# la latencia, el jitter y la perdida:
#   - critico:    VoIP y Videoconferencia (tiempo real, no toleran cortes)
#   - normal:     Web educativa / plataforma de aprendizaje (interactivo, tolerante)
#   - background: Streaming, transferencia de archivos, juegos y otros (volumen,
#                 tolerantes a la demora porque usan bufer)
# -----------------------------------------------------------------------------
CLASS_TO_SERVICES = {
    'critico':    ['VoIP', 'Videoconferencia'],
    'normal':     ['Web_Educativa'],
    'background': ['Streaming', 'TransferenciaArchivos', 'Juegos', 'Otros'],
}

# -----------------------------------------------------------------------------
# TRADUCCION DE CADA CLASE DE PRIORIDAD A MARCA DSCP Y A COLA HTB
# La decision del modelo (3 clases) se materializa de dos formas complementarias:
#   1) Se marca el paquete con un valor DSCP estandar (observabilidad y trato
#      diferenciado extremo a extremo).
#   2) Se encola el flujo en la cola HTB correspondiente del router (priorizacion
#      efectiva del ancho de banda).
# Las marcas DSCP que aparecen en la telemetria son CONSECUENCIA de esta
# traduccion de las tres clases, no clases independientes.
#   critico    -> DSCP EF (46)              -> cola HTB 1:10 (prioridad alta)
#   normal     -> DSCP AF31 (26)            -> cola HTB 1:20 (prioridad media)
#   background -> DSCP AF22/AF11/BE (segun subtipo) -> cola HTB 1:30 (prioridad baja)
# -----------------------------------------------------------------------------
# Marca DSCP principal por clase (decision del Random Forest):
#   critico    -> EF    | normal -> AF31 | background -> se refina por subtipo (ver abajo)
CLASS_TO_DSCP = {0: 'EF', 1: 'AF31', 2: 'BE'}

# Cola HTB por clase (priorizacion efectiva del ancho de banda en el router):
CLASS_TO_HTB  = {0: '1:10', 1: '1:20', 2: '1:30'}

# -----------------------------------------------------------------------------
# SEGUNDA CAPA: REFINAMIENTO DEL TRAFICO BACKGROUND POR SUBTIPO DE APLICACION
# El Random Forest decide tres clases de prioridad. La clase 'background' agrupa
# servicios distintos (streaming, transferencia de archivos, juegos y otros). El
# marcado DSCP fino de estos subtipos NO lo hace el modelo, sino una segunda capa
# de clasificacion por aplicacion ubicada en el firewall perimetral (FortiGate) y
# en el conmutador, que refina el marcado dentro de la cola de baja prioridad:
#   Streaming        -> DSCP AF22  (video diferido, tolerante a buffer)
#   TransferenciaArch-> DSCP AF11  (descargas masivas, maxima tolerancia a demora)
#   Juegos / Otros   -> DSCP BE    (mejor esfuerzo)
# Esta capa explica por que la telemetria registra cinco marcas DSCP (EF, AF31,
# AF22, AF11, BE) aunque el modelo solo decida tres clases de prioridad: las dos
# marcas adicionales (AF22 y AF11) son el resultado del refinamiento por subtipo
# dentro de la clase background, no clases independientes del modelo.
# Todos los subtipos de background comparten la MISMA cola HTB de baja prioridad
# (1:30); el refinamiento DSCP solo afecta el marcado, no la cola.
# -----------------------------------------------------------------------------
def refine_background_dscp(features):
    """Refina la marca DSCP dentro de la clase background segun el subtipo.
    Recibe el vector de 10 features y devuelve la marca DSCP del subtipo.
    Usa el throughput (bytes_per_second) y el tamano de paquete para distinguir:
      - Transferencia de archivos: paquetes grandes (MTU) y throughput muy alto -> AF11
      - Streaming: throughput alto sostenido pero paquetes algo menores         -> AF22
      - Juegos / Otros: throughput bajo o intermitente                          -> BE
    Los umbrales se calibraron sobre el trafico real de la clase background."""
    # Indices de las features dentro del vector (ver FEATURE_NAMES)
    packet_size = features[0]
    bytes_per_second = features[9]

    # Transferencia de archivos: paquetes de tamano MTU completo y throughput muy alto
    if packet_size >= 1300 and bytes_per_second >= 1426000:
        return 'AF11'
    # Streaming: throughput alto pero no tan extremo como una descarga masiva
    elif bytes_per_second >= 1341000:
        return 'AF22'
    # Juegos y otros: el resto del background
    else:
        return 'BE'

# Las 10 caracteristicas (features) que el modelo analiza de cada flujo de trafico.
# Son las "pistas" que recibe el modelo (extraidas de la captura de paquetes) para
# decidir la clase de prioridad. El agente NO captura paquetes: recibe estas 10
# features ya calculadas por el proceso de captura (ver nota al final del archivo).
FEATURE_NAMES = [
    'packet_size',           # Tamano promedio de los paquetes (bytes)
    'inter_arrival_time',    # Tiempo entre paquetes (ms)
    'jitter',                # Variacion del tiempo entre paquetes (ms)
    'packet_loss_rate',      # Porcentaje de paquetes que se perdieron
    'bandwidth_utilization', # Uso del ancho de banda (%)
    'rtp_payload_type',      # Tipo de contenido RTP (96 = H264 video, 0 = audio/no-RTP)
    'burst_duration',        # Duracion de rafagas de paquetes (ms)
    'flow_duration',         # Duracion total del flujo (segundos)
    'packets_per_second',    # Tasa de paquetes (pps)
    'bytes_per_second'       # Throughput (bytes por segundo)
]

# Ventana deslizante: guarda las ultimas 60 mediciones (cuando llega la 61, elimina la 1)
# Sirve para calcular promedios recientes sin usar toda la historia
metrics_window = deque(maxlen=60)


# =============================================================================
# FUNCION: estimate_mos
# Calcula el MOS (Mean Opinion Score) en escala de 1 a 5.
#
# IMPORTANTE: esta funcion NO implementa la norma ITU-T P.800 ni es una
# implementacion completa de ITU-T P.1203. Es un MODELO PARAMETRICO PROPIO,
# inspirado en los factores de degradacion que la Recomendacion ITU-T P.1203
# identifica como relevantes (latencia, jitter, perdida, estabilidad de la
# reproduccion y resolucion). Los pesos y umbrales fueron definidos y calibrados
# para el contexto de la red de la academia.
# =============================================================================
def estimate_mos(latency, jitter, loss, buffering, resolution):
    """Modelo parametrico propio inspirado en los factores de ITU-T P.1203.
    Funcion de coste: J = alpha*L + beta*J + gamma*P + delta*B - lambda*R"""
    # Se normaliza cada metrica a un valor entre 0 y 1 (0 = perfecto, 1 = pesimo)
    # Los umbrales se eligieron como limites operativos para servicios en tiempo real:

    # Latencia: se considera pesima a partir de 400 ms (limite para videoconferencia)
    L_n = min(latency / 400.0, 1.0)

    # Jitter: se considera pesimo a partir de 100 ms
    J_n = min(jitter / 100.0, 1.0)

    # Perdida: se considera pesima a partir de 20%
    P_n = min(loss / 20.0, 1.0)

    # Buffering: pesimo a partir de 15 eventos/minuto
    B_n = min(buffering / 15.0, 1.0)

    # Resolucion: aqui si queremos que sea ALTA (por eso se RESTA al coste)
    R_n = resolution / 100.0

    # Pesos de cada factor (los cuatro de degradacion suman 1.0; la resolucion
    # se resta como factor positivo). Calibrados para la red de la academia,
    # tomando como referencia la importancia relativa que ITU-T P.1203 asigna a
    # cada factor:
    #   alpha=0.25 latencia | beta=0.20 jitter | gamma=0.25 perdida
    #   delta=0.15 buffering | lambda=0.15 resolucion (positiva, se resta)
    alpha, beta, gamma, delta, lam = 0.25, 0.20, 0.25, 0.15, 0.15

    # Formula del coste: mas coste = peor QoE
    cost = alpha*L_n + beta*J_n + gamma*P_n + delta*B_n - lam*R_n

    # MOS = 5 - 4*cost, limitado entre 1 (muy malo) y 5 (excelente)
    return round(max(1.0, min(5.0, 5.0 - 4.0 * cost)), 2)


# =============================================================================
# FUNCION: cold_start_model  (antes "_bootstrap_model")
# Crea un modelo PROVISIONAL con datos sinteticos SOLO para arranque en frio.
#
# ADVERTENCIA: este NO es el modelo cientifico de la tesis. Se genera unicamente
# para que el servicio web no caiga si arranca sin un modelo entrenado en disco.
# El modelo cientifico se entrena con trafico REAL via el endpoint /train. Este
# modelo provisional queda marcado en sus metadatos como 'cold_start_sintetico'
# para que nunca se confunda con el modelo real ni se reporte su OOB en la tesis.
# =============================================================================
def cold_start_model():
    global model, model_trained, model_meta

    logger.warning('ARRANQUE EN FRIO: generando modelo PROVISIONAL sintetico. '
                   'Este modelo NO es el modelo cientifico. Reentrene con datos '
                   'reales via /train antes de usar en produccion.')

    # Semilla fija para que el arranque sea reproducible
    np.random.seed(42)

    # Numero de muestras por cada clase (solo para el arranque en frio)
    n = 300

    # ------- CLASE 0: CRITICO (VoIP y videoconferencia, RTP en tiempo real) -------
    X0 = np.column_stack([
        np.random.normal(800,100,n),    # packet_size: ~800 bytes (paquetes RTP tipicos)
        np.random.normal(20,5,n),       # inter_arrival_time: ~20ms (50 pps para video)
        np.random.normal(5,2,n),        # jitter: bajo (~5ms)
        np.random.normal(1,0.5,n),      # packet_loss_rate: muy baja (~1%)
        np.random.normal(60,10,n),      # bandwidth_utilization: 60% (video HD)
        np.ones(n)*96,                  # rtp_payload_type: 96 (H264)
        np.random.normal(200,50,n),     # burst_duration: 200ms
        np.random.normal(300,60,n),     # flow_duration: 300s (5 min de clase)
        np.random.normal(50,10,n),      # packets_per_second: ~50
        np.random.normal(400000,50000,n)# bytes_per_second: ~400 KB/s (video HD)
    ])

    # ------- CLASE 1: NORMAL (web educativa / plataforma de aprendizaje) -------
    X1 = np.column_stack([
        np.random.normal(500,200,n),    # packet_size: variable (~500 bytes)
        np.random.normal(80,30,n),      # inter_arrival_time: mayor (respuestas HTTP)
        np.random.normal(15,8,n),       # jitter: medio
        np.random.normal(3,1,n),        # packet_loss: un poco mas alta
        np.random.normal(30,15,n),      # bandwidth: menor uso
        np.zeros(n),                    # no es RTP (payload_type = 0)
        np.random.normal(100,40,n),     # burst corto
        np.random.normal(150,50,n),     # flujos cortos (requests HTTP)
        np.random.normal(20,8,n),       # pocas pps
        np.random.normal(250000,100000,n) # throughput variable
    ])

    # ------- CLASE 2: BACKGROUND (streaming, descargas, juegos, otros) -------
    X2 = np.column_stack([
        np.random.normal(1400,50,n),    # packet_size: ~1400 (MTU completo, descargas)
        np.random.normal(5,2,n),        # inter_arrival_time: muy bajo (rafaga)
        np.random.normal(2,1,n),        # jitter: bajo (TCP bien gestionado)
        np.random.normal(0.5,0.2,n),    # packet_loss: baja (TCP retransmite)
        np.random.normal(80,10,n),      # bandwidth: alto (80% - consume todo)
        np.zeros(n),                    # no RTP
        np.random.normal(500,100,n),    # rafagas largas
        np.random.normal(600,100,n),    # flujos largos (descarga de archivos)
        np.random.normal(100,20,n),     # muchas pps
        np.random.normal(1400000,100000,n) # throughput alto (~1.4 MB/s)
    ])

    # Se unen las 3 matrices verticalmente: X = [X0; X1; X2] -> 900 filas x 10 columnas
    X = np.vstack([X0, X1, X2])

    # Vector de etiquetas: 300 ceros (critico) + 300 unos (normal) + 300 doses (background)
    y = np.array([0]*n + [1]*n + [2]*n)

    # Se crea el Random Forest con el MISMO numero de arboles que el modelo real (100)
    m = RandomForestClassifier(n_estimators=N_ESTIMATORS, oob_score=True,
                               random_state=42, n_jobs=-1)
    m.fit(X, y)

    # Guarda el modelo provisional en disco
    joblib.dump(m, MODEL_PATH)

    # Guarda los metadatos marcandolo claramente como sintetico de arranque en frio
    model_meta = {
        'origen': 'cold_start_sintetico',
        'advertencia': 'Modelo provisional de arranque. NO es el modelo cientifico. '
                       'Reentrenar con datos reales via /train.',
        'n_estimators': N_ESTIMATORS,
        'n_muestras': int(len(X)),
        'oob_error': round(1 - m.oob_score_, 4),
        'fecha': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    _save_meta(model_meta)

    model, model_trained = m, True

    # El OOB del arranque en frio se publica para monitoreo del servicio, pero
    # NO debe reportarse como resultado cientifico (es sobre datos sinteticos).
    model_oob_gauge.set(model_meta['oob_error'])
    logger.warning('Modelo de arranque en frio listo (sintetico). OOB=%.4f. '
                   'Pendiente reentrenar con datos reales.', model_meta['oob_error'])


# =============================================================================
# FUNCIONES AUXILIARES PARA METADATOS DEL MODELO
# =============================================================================
def _save_meta(meta):
    """Guarda los metadatos del modelo en disco como evidencia."""
    try:
        os.makedirs(os.path.dirname(META_PATH), exist_ok=True)
        with open(META_PATH, 'w') as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        logger.debug('No se pudo guardar meta: %s', e)


def _load_meta():
    """Carga los metadatos del modelo desde disco, si existen."""
    try:
        if os.path.exists(META_PATH):
            with open(META_PATH) as f:
                return json.load(f)
    except Exception as e:
        logger.debug('No se pudo leer meta: %s', e)
    return {}


# =============================================================================
# FUNCION: load_model
# Al arrancar el servicio, intenta cargar el modelo desde disco.
# Si no existe, crea uno provisional con cold_start_model().
# =============================================================================
def load_model():
    global model, model_trained, model_meta

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        model_trained = True
        model_meta = _load_meta()

        if hasattr(model, 'oob_score_'):
            model_oob_gauge.set(1 - model.oob_score_)

        origen = model_meta.get('origen', 'desconocido')
        logger.info('Modelo RF cargado desde disco. Origen=%s', origen)
        if origen == 'cold_start_sintetico':
            logger.warning('El modelo cargado es el de arranque en frio (sintetico). '
                           'Reentrene con datos reales via /train para produccion.')
    else:
        logger.warning('Sin modelo guardado en disco, generando arranque en frio...')
        cold_start_model()


# =============================================================================
# FUNCION: apply_tc_rule
# Aplica la accion de QoS al router de Linux. La decision del modelo (3 clases)
# se materializa en (1) una marca DSCP y (2) el encolado en la cola HTB.
# =============================================================================
def apply_tc_rule(traffic_class, interface='eth0', features=None):
    # Cola HTB destino segun la clase: critico->1:10, normal->1:20, background->1:30
    htb = CLASS_TO_HTB.get(traffic_class, '1:20')

    # Marca DSCP. Para critico y normal es directa (EF / AF31). Para background,
    # la segunda capa refina por subtipo de aplicacion (AF22 / AF11 / BE).
    if traffic_class == 2 and features is not None:
        dscp = refine_background_dscp(features)
    else:
        dscp = CLASS_TO_DSCP.get(traffic_class, 'AF31')

    # Comando tc que envia el flujo a la cola HTB correspondiente del router.
    # - dev eth0: interfaz de red de salida
    # - protocol ip: solo IPv4
    # - prio N: prioridad del filtro (1=mayor, 3=menor)
    # - u32 match ip protocol 17: protocolo 17 = UDP (RTP/video en tiempo real)
    # - flowid 1:N0: cola HTB destino
    cmd = (
        f"tc filter add dev {interface} protocol ip prio {traffic_class+1} "
        f"u32 match ip protocol 17 0xff flowid {htb} 2>/dev/null || true"
    )

    try:
        subprocess.run(cmd, shell=True, timeout=2)
        tc_rules_applied.labels(traffic_class=TRAFFIC_CLASSES[traffic_class]).inc()
        logger.debug('Clase=%s -> DSCP=%s (refinado si background) -> cola HTB=%s',
                     TRAFFIC_CLASSES[traffic_class], dscp, htb)
    except Exception as e:
        logger.debug('tc error: %s', e)


# =============================================================================
# ENDPOINTS DE LA API REST (FLASK)
# =============================================================================

@app.route('/')
def index():
    return jsonify({
        'service': 'Agente ML - Random Forest QoE',
        'project': 'Tesis SQUAD - Bogota 2026',
        'status': 'running',
        'model_trained': model_trained,
        'model_origin': model_meta.get('origen', 'desconocido'),
        'classes': TRAFFIC_CLASSES,
        'class_to_services': CLASS_TO_SERVICES,
        'endpoints': {
            'GET  /health':        'Estado del servicio',
            'GET  /model/info':    'Informacion del modelo RF',
            'POST /predict':       'Clasificar flujo de trafico',
            'POST /qoe':           'Actualizar metricas QoE',
            'GET  /qoe/snapshot':  'Snapshot de metricas recientes',
            'POST /train':         'Reentrenar modelo con CSV real',
            'GET  /metrics':       'Metricas Prometheus'
        }
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_trained': model_trained,
                    'model_origin': model_meta.get('origen', 'desconocido')})


# =============================================================================
# ENDPOINT /predict - El mas importante
# Recibe las 10 features de un flujo y devuelve la clase de prioridad predicha
# =============================================================================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Valida que venga el campo 'features' (debe ser un array de 10 numeros)
    if not data or 'features' not in data:
        return jsonify({'error': 'Se requiere campo features (10 valores)'}), 400

    feats = data['features']
    # Valida que sean exactamente 10 features (las que el modelo espera)
    if len(feats) != len(FEATURE_NAMES):
        return jsonify({'error': f'Se requieren {len(FEATURE_NAMES)} features, '
                                 f'se recibieron {len(feats)}'}), 400

    t0 = time.time()

    # Convierte la lista en un arreglo numpy 2D de forma (1, 10)
    features = np.array(feats).reshape(1, -1)

    # El modelo predice la clase de prioridad (0=critico, 1=normal, 2=background)
    pred_class = int(model.predict(features)[0])
    pred_proba = model.predict_proba(features)[0].tolist()

    elapsed_ms = (time.time() - t0) * 1000
    inference_latency.observe(elapsed_ms)
    inference_counter.inc()
    packets_classified.labels(traffic_class=TRAFFIC_CLASSES[pred_class]).inc()

    # Aplica la accion: marca DSCP (refinada si es background) + cola HTB
    feat_list = features.flatten().tolist()
    apply_tc_rule(pred_class, features=feat_list)

    # Marca DSCP efectiva (refinada por subtipo si la clase es background)
    if pred_class == 2:
        dscp_efectiva = refine_background_dscp(feat_list)
    else:
        dscp_efectiva = CLASS_TO_DSCP[pred_class]

    return jsonify({
        'class': pred_class,
        'class_name': TRAFFIC_CLASSES[pred_class],
        'services_in_class': CLASS_TO_SERVICES[TRAFFIC_CLASSES[pred_class]],
        'dscp_mark': dscp_efectiva,
        'htb_queue': CLASS_TO_HTB[pred_class],
        'probabilities': {TRAFFIC_CLASSES[i]: round(p, 4) for i, p in enumerate(pred_proba)},
        'inference_ms': round(elapsed_ms, 3)
    })


# =============================================================================
# ENDPOINT /qoe - Actualiza las metricas QoE medidas por el sistema de captura
# =============================================================================
@app.route('/qoe', methods=['POST'])
def update_qoe():
    data = request.get_json()
    lat = data.get('latency_ms', 0)
    jit = data.get('jitter_ms', 0)
    loss = data.get('packet_loss_pct', 0)
    buf = data.get('buffering_events_per_min', 0)
    res = data.get('resolution_optimal_pct', 100)

    metrics_window.append({'latency': lat, 'jitter': jit, 'loss': loss,
                            'buffering': buf, 'resolution': res, 'ts': time.time()})

    qoe_latency_gauge.set(lat)
    qoe_jitter_gauge.set(jit)
    qoe_loss_gauge.set(loss)
    qoe_buffering_gauge.set(buf)
    qoe_resolution_gauge.set(res)

    mos = estimate_mos(lat, jit, loss, buf, res)
    qoe_mos_gauge.set(mos)

    return jsonify({'mos': mos})


# =============================================================================
# ENDPOINT /qoe/snapshot - Estadisticas de las ultimas 60 mediciones
# =============================================================================
@app.route('/qoe/snapshot')
def qoe_snapshot():
    if not metrics_window:
        return jsonify({'error': 'Sin datos aun'}), 404

    df = pd.DataFrame(list(metrics_window))
    return jsonify({
        'latency_ms':         {'mean': round(df['latency'].mean(), 2),   'std': round(df['latency'].std(), 2)},
        'jitter_ms':          {'mean': round(df['jitter'].mean(), 2),    'std': round(df['jitter'].std(), 2)},
        'packet_loss_pct':    {'mean': round(df['loss'].mean(), 2),      'std': round(df['loss'].std(), 2)},
        'buffering_per_min':  {'mean': round(df['buffering'].mean(), 2), 'std': round(df['buffering'].std(), 2)},
        'resolution_opt_pct': {'mean': round(df['resolution'].mean(), 2),'std': round(df['resolution'].std(), 2)},
        'mos_estimated': estimate_mos(df['latency'].mean(), df['jitter'].mean(),
                                      df['loss'].mean(), df['buffering'].mean(),
                                      df['resolution'].mean()),
        'samples': len(df)
    })


# =============================================================================
# ENDPOINT /train - Entrena/reentrena el modelo con un CSV de trafico REAL
#
# Este es el endpoint que produce el MODELO CIENTIFICO de la tesis. El CSV debe
# contener las 10 features y la columna 'label' (0=critico, 1=normal, 2=background)
# con trafico real capturado y etiquetado de la red. El OOB que devuelve este
# endpoint es el que se reporta en la tesis, y el CSV debe conservarse como
# evidencia del entrenamiento.
# =============================================================================
@app.route('/train', methods=['POST'])
def train():
    if 'file' not in request.files:
        return jsonify({'error': 'Se requiere CSV con trafico real'}), 400

    df = pd.read_csv(request.files['file'])

    # Valida que esten las 10 features y la etiqueta
    faltan = [c for c in FEATURE_NAMES if c not in df.columns]
    if faltan:
        return jsonify({'error': f'Faltan columnas: {faltan}'}), 400
    if 'label' not in df.columns:
        return jsonify({'error': "Falta la columna 'label'"}), 400

    X = df[FEATURE_NAMES].values
    y = df['label'].values

    # Se usa el MISMO numero de arboles que el modelo desplegado (100) para que
    # el modelo entrenado y el reportado en la tesis coincidan en este parametro.
    rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, oob_score=True,
                                random_state=42, n_jobs=-1)
    rf.fit(X, y)
    joblib.dump(rf, MODEL_PATH)

    global model, model_trained, model_meta
    model, model_trained = rf, True

    oob_err = 1 - rf.oob_score_
    model_oob_gauge.set(oob_err)

    # Marca el modelo como entrenado con datos REALES (modelo cientifico)
    model_meta = {
        'origen': 'entrenado_datos_reales',
        'n_estimators': N_ESTIMATORS,
        'n_muestras': int(len(X)),
        'oob_error': round(oob_err, 4),
        'oob_score': round(rf.oob_score_, 4),
        'fecha': time.strftime('%Y-%m-%d %H:%M:%S'),
        'feature_importance': dict(zip(FEATURE_NAMES, rf.feature_importances_.round(4).tolist())),
    }
    _save_meta(model_meta)

    return jsonify({
        'status': 'trained',
        'origen': 'entrenado_datos_reales',
        'oob_error': round(oob_err, 4),
        'oob_score': round(rf.oob_score_, 4),
        'n_samples': len(X),
        'feature_importance': model_meta['feature_importance']
    })


# =============================================================================
# ENDPOINT /model/info - Informacion sobre el estado actual del modelo
# =============================================================================
@app.route('/model/info')
def model_info():
    info = {'trained': model_trained,
            'type': type(model).__name__ if model else None,
            'origin': model_meta.get('origen', 'desconocido'),
            'n_estimators': N_ESTIMATORS,
            'features': FEATURE_NAMES,
            'n_features': len(FEATURE_NAMES),
            'classes': TRAFFIC_CLASSES,
            'class_to_services': CLASS_TO_SERVICES,
            'class_to_dscp': CLASS_TO_DSCP,
            'class_to_htb': CLASS_TO_HTB,
            'meta': model_meta}

    if model_trained and hasattr(model, 'oob_score_'):
        info['oob_score'] = round(model.oob_score_, 4)
        info['oob_error'] = round(1 - model.oob_score_, 4)

    return jsonify(info)


# =============================================================================
# ENDPOINT /metrics - Expone todas las metricas en formato Prometheus
# =============================================================================
@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


# =============================================================================
# PUNTO DE ENTRADA DEL PROGRAMA
# =============================================================================
if __name__ == '__main__':
    logger.info('Iniciando Agente ML - Tesis QoE SQUAD Bogota 2026')
    load_model()
    app.run(host='0.0.0.0', port=5001, threaded=True)


# =============================================================================
# NOTA SOBRE LA EXTRACCION DE CARACTERISTICAS (de donde salen las 10 features)
# =============================================================================
# Este agente NO captura paquetes directamente. Recibe por HTTP (endpoint
# /predict) las 10 features ya calculadas, y SOLO opera durante el segmento final
# del estudio, cuando el modelo esta activo en produccion. Durante el segmento
# inicial el agente no clasifica trafico: en ese periodo la red opera con sus
# politicas estaticas y la medicion de la calidad se realiza de forma
# independiente mediante captura de paquetes con un analizador externo, sin
# intervencion de este agente.
#
# El proceso de extraccion de caracteristicas se realiza por fuera de este
# archivo: agrega los paquetes por flujo y calcula las 10 features (tamano medio,
# tiempo entre paquetes, jitter, perdida, uso de ancho de banda, tipo de payload
# RTP, duracion de rafaga, duracion del flujo, paquetes por segundo y bytes por
# segundo) que aqui se reciben y se clasifican.
# =============================================================================
