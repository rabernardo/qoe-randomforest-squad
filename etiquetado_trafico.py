"""
================================================================================
SCRIPT DE ETIQUETADO DE TRAFICO - DATASET DE ENTRENAMIENTO RANDOM FOREST
Tesis: Optimizacion de la QoE mediante Random Forest - Academia SQUAD, Bogota 2026

Este script asigna la etiqueta de clase (critico, normal o background) a cada
flujo de trafico capturado, mediante reglas de identificacion por puerto,
protocolo y servicio destino. Es el procedimiento con el que se construyo el
conjunto de entrenamiento del modelo (las muestras etiquetadas con las 10
caracteristicas y su clase).

El etiquetado NO es manual ni arbitrario: se basa en el conocimiento previo de
los servicios que operan en la red de la academia (que puerto y protocolo usa
cada servicio), lo que permite asignar la clase de forma objetiva y reproducible.

NOTA DE PRIVACIDAD: las direcciones IP de los servidores institucionales se han
reemplazado por direcciones de ejemplo del rango de documentacion (RFC 5737,
192.0.2.0/24). En el entorno real se sustituyen por las IP de los servidores de
la institucion. La logica del etiquetado no cambia.

ENTRADA:  un CSV con los flujos capturados (una fila por flujo) que incluya, ademas
          de las 10 caracteristicas, los campos de identificacion del flujo:
          protocol_l4 (TCP/UDP), dst_port, rtp_payload_type, dst_ip (opcional).
SALIDA:   el mismo CSV con dos columnas anadidas: 'label' (0/1/2) y 'label_name'.
================================================================================
"""

import sys
import pandas as pd

# =============================================================================
# TABLA DE CLASES
# =============================================================================
# 0 = critico    : servicios en tiempo real (VoIP, videoconferencia)
# 1 = normal     : servicios interactivos tolerantes (web educativa, LMS)
# 2 = background : servicios de volumen tolerantes a demora (streaming, descargas,
#                  juegos, otros)
CLASES = {0: 'critico', 1: 'normal', 2: 'background'}


# =============================================================================
# REGLAS DE IDENTIFICACION DE SERVICIOS DE LA RED DE LA ACADEMIA
# Cada servicio se identifica por su puerto, protocolo o servidor destino. Estos
# valores corresponden a la configuracion real de los servicios de la academia.
# Las IP de los servidores se definen como parametros de configuracion; aqui se
# usan IP de ejemplo (RFC 5737, 192.0.2.0/24).
# =============================================================================

# --- CRITICO: VoIP y Videoconferencia (tiempo real, RTP/UDP) ---
# Jitsi (videoconferencia) usa UDP en el rango 10000-20000 para los flujos RTP.
# VoIP/SIP usa los puertos 5060/5061. El payload RTP 96 corresponde a video H264;
# los payloads RTP de audio (0, 8, 9, 18) tambien son tiempo real.
PUERTOS_VIDEOCONF_UDP = range(10000, 20001)   # rango de medios de Jitsi
PUERTOS_VOIP         = {5060, 5061, 3478}      # SIP y STUN/TURN
RTP_PAYLOADS_TIEMPO_REAL = {0, 8, 9, 18, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107}

# --- NORMAL: Web educativa / plataforma de aprendizaje (HTTP/HTTPS al LMS) ---
# El trafico web educativo va por los puertos 80/443 hacia el servidor LMS.
PUERTOS_WEB = {80, 443, 8080, 8443}
# Servidores institucionales (LMS, ERP) -> trafico normal.
# IP de ejemplo (RFC 5737); reemplazar por las IP reales en el entorno productivo.
LMS_SERVER_IP = '192.0.2.11'
ERP_SERVER_IP = '192.0.2.12'
IPS_NORMALES = {LMS_SERVER_IP, ERP_SERVER_IP}

# --- BACKGROUND: streaming, transferencia de archivos, juegos, otros ---
# Streaming de video por Internet (servicios de video bajo demanda)
PUERTOS_STREAMING = {1935, 554}                 # RTMP, RTSP
# Transferencia de archivos (descargas masivas)
PUERTOS_FILE = {21, 22, 445, 2049, 873}         # FTP, SFTP, SMB, NFS, rsync
# Servidor de archivos institucional.
# IP de ejemplo (RFC 5737); reemplazar por la IP real en el entorno productivo.
FILE_SERVER_IP = '192.0.2.13'
IPS_FILE = {FILE_SERVER_IP}
# Juegos en linea (rangos tipicos de puertos altos UDP)
PUERTOS_GAMING = {3074, 3075, 27015, 27036}     # puertos comunes de gaming


# =============================================================================
# FUNCION DE ETIQUETADO POR FLUJO
# Aplica las reglas en orden de prioridad: primero critico, luego normal, y todo
# lo que no encaje cae en background (mejor esfuerzo, que es el comportamiento por
# defecto de la red para trafico no clasificado como prioritario).
# =============================================================================
def etiquetar_flujo(fila):
    proto = str(fila.get('protocol_l4', '')).upper()
    dst_port = int(fila.get('dst_port', 0) or 0)
    payload = int(fila.get('rtp_payload_type', 0) or 0)
    dst_ip = str(fila.get('dst_ip', '') or '')

    # ----- REGLA 1: CRITICO (tiempo real) -----
    # Videoconferencia: UDP en el rango de medios de Jitsi con payload RTP valido
    if proto == 'UDP' and dst_port in PUERTOS_VIDEOCONF_UDP and payload in RTP_PAYLOADS_TIEMPO_REAL:
        return 0
    # VoIP/SIP
    if dst_port in PUERTOS_VOIP:
        return 0
    # Flujo RTP con payload de tiempo real aunque el puerto no caiga en el rango
    if proto == 'UDP' and payload in RTP_PAYLOADS_TIEMPO_REAL and payload != 0:
        return 0

    # ----- REGLA 2: NORMAL (web educativa interactiva) -----
    # Trafico web hacia los servidores institucionales (LMS/ERP)
    if dst_port in PUERTOS_WEB and (dst_ip in IPS_NORMALES or dst_ip == ''):
        return 1
    # Trafico web general TCP a puertos web
    if proto == 'TCP' and dst_port in PUERTOS_WEB:
        return 1

    # ----- REGLA 3: BACKGROUND (todo lo demas) -----
    # Streaming, transferencia de archivos, juegos y cualquier flujo no prioritario
    # caen en background por defecto.
    return 2


# =============================================================================
# FUNCION PRINCIPAL
# =============================================================================
def etiquetar_csv(ruta_entrada, ruta_salida):
    df = pd.read_csv(ruta_entrada)

    # Verifica que existan los campos de identificacion necesarios
    requeridos = ['protocol_l4', 'dst_port', 'rtp_payload_type']
    faltan = [c for c in requeridos if c not in df.columns]
    if faltan:
        print(f'ADVERTENCIA: faltan columnas de identificacion: {faltan}')
        print('El etiquetado usara solo los campos disponibles.')

    # Aplica la regla de etiquetado a cada flujo
    df['label'] = df.apply(etiquetar_flujo, axis=1)
    df['label_name'] = df['label'].map(CLASES)

    df.to_csv(ruta_salida, index=False)

    # Resumen del etiquetado
    print(f'Etiquetado completado: {len(df)} flujos')
    print('Distribucion por clase:')
    for clase, nombre in CLASES.items():
        n = int((df['label'] == clase).sum())
        pct = 100 * n / len(df) if len(df) else 0
        print(f'  {nombre:12} (clase {clase}): {n:5d} flujos ({pct:.1f}%)')
    print(f'Archivo etiquetado guardado en: {ruta_salida}')
    return df


# =============================================================================
# PUNTO DE ENTRADA
# Uso:  python etiquetado_trafico.py  flujos_capturados.csv  flujos_etiquetados.csv
# =============================================================================
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Uso: python etiquetado_trafico.py <entrada.csv> <salida.csv>')
        sys.exit(1)
    etiquetar_csv(sys.argv[1], sys.argv[2])
