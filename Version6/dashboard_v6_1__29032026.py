import cv2
import numpy as np
import pandas as pd
from collections import deque
from ultralytics import YOLO

print("--- INICIANDO SISTEMA V6.1: HÍBRIDO + CINEMÁTICA + MEMORIA DE ZONA ---")

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
path_modelo = r"C:\Users\ivang\Desktop\Tesis_Hockey\models\best_v5.pt" 
path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\6taFecha-2024-Vélez _B_ 2-0DAOM.mp4" 

VIDEO_W, VIDEO_H = 800, 600

# ==========================================
# 2. MOTOR ANALÍTICO Y CINEMÁTICA VERTICAL
# ==========================================
metricas_recuperacion = {
    "Z1_ArcoLocal_25yd": 0,
    "Z2_25yd_50yd_Local": 0,
    "Z3_50yd_25yd_Visita": 0,
    "Z4_25yd_ArcoVisita": 0
}

historial_flujo_y = deque(maxlen=15) 
estado_posesion = "Indefinido" 
registro_eventos = [] 

# --- NUEVAS VARIABLES DE CONTROL TÁCTICO ---
ultima_zona_valida = "Z2_25yd_50yd_Local" # Valor por defecto seguro
frames_cambio_estado = 0
UMBRAL_FLUJO = 0.6  # Umbral más bajo para captar paneos de cámara más suaves
FRAMES_CONFIRMACION = 5 # Debe mantener el paneo 5 frames para contar como Turnover

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
color_gris_previo = None
puntos_previos = None

# ==========================================
# 3. REPRODUCTOR DE VIDEO (Controles)
# ==========================================
estado_reproduccion = "PLAY"
salto_solicitado = True 
cap = cv2.VideoCapture(path_video)
fps_video = cap.get(cv2.CAP_PROP_FPS) or 30

def click_botones(event, x, y, flags, param):
    global estado_reproduccion, salto_solicitado
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if 550 <= y <= 590:
            if 10 <= x <= 80:    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_actual - 5 * fps_video)); salto_solicitado = True
            elif 90 <= x <= 160: cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_actual - 1 * fps_video)); salto_solicitado = True
            elif 170 <= x <= 270: estado_reproduccion = "PAUSE" if estado_reproduccion == "PLAY" else "PLAY"
            elif 280 <= x <= 350: cap.set(cv2.CAP_PROP_POS_FRAMES, frame_actual + 1 * fps_video); salto_solicitado = True
            elif 360 <= x <= 430: cap.set(cv2.CAP_PROP_POS_FRAMES, frame_actual + 5 * fps_video); salto_solicitado = True

# ==========================================
# 4. FUNCIONES GEOMÉTRICAS
# ==========================================
def obtener_centroide(xyxy):
    return (int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2))

def extraer_linea_hough(frame, box_xyxy):
    try:
        alto_frame, ancho_frame = frame.shape[:2]
        x1, y1 = max(0, int(box_xyxy[0])), max(0, int(box_xyxy[1]))
        x2, y2 = min(ancho_frame, int(box_xyxy[2])), min(alto_frame, int(box_xyxy[3]))
        if x2 <= x1 or y2 <= y1: return None
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return None
            
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mascara_blanca = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([179, 50, 255]))
        edges = cv2.Canny(mascara_blanca, 30, 100) 
        lineas = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=40, maxLineGap=20)
        
        if lineas is not None and len(lineas) > 0:
            linea_mas_larga = max(lineas, key=lambda l: (l[0][2]-l[0][0])**2 + (l[0][3]-l[0][1])**2)[0]
            xl1, yl1, xl2, yl2 = linea_mas_larga
            vx, vy = xl2 - xl1, yl2 - yl1
            norma = np.sqrt(vx**2 + vy**2)
            if norma != 0: return (float(vx/norma), float(vy/norma), float(xl1 + x1), float(yl1 + y1))
        return None
    except Exception: return None

def calcular_interseccion(recta1, recta2):
    if recta1 is None or recta2 is None: return None
    vx1, vy1, x1, y1 = recta1; vx2, vy2, x2, y2 = recta2
    det = (vx1 * vy2) - (vy1 * vx2)
    if abs(det) < 1e-6: return None
    t1 = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det
    return (int(x1 + t1 * vx1), int(y1 + t1 * vy1))

def inferir_zona_horizonte(objetos_yolo):
    for obj in objetos_yolo:
        if obj["clase"] == "goal":
            return "Z1_ArcoLocal_25yd" if obj["cy"] > (VIDEO_H / 2) else "Z4_25yd_ArcoVisita"
    for obj in objetos_yolo:
        if obj["clase"] == "25yd line":
            return "Z2_25yd_50yd_Local" if obj["cy"] > (VIDEO_H / 2) else "Z3_50yd_25yd_Visita"
    return "Zona_Transicion"

# ==========================================
# 5. INICIALIZACIÓN
# ==========================================
cv2.namedWindow("Dashboard Analitico V6.1")
cv2.setMouseCallback("Dashboard Analitico V6.1", click_botones)
model = YOLO(path_modelo)
annotated_frame = None

# ==========================================
# 6. BUCLE PRINCIPAL
# ==========================================
while cap.isOpened():
    if estado_reproduccion == "PLAY" or salto_solicitado:
        success, frame = cap.read()
        if not success: break
        
        # --- PROTECCIÓN DEL REPRODUCTOR ---
        # Si el usuario saltó en el tiempo, reseteamos el flujo óptico para que no cuente como recuperación
        if salto_solicitado:
            color_gris_previo = None
            puntos_previos = None
            historial_flujo_y.clear()
            frames_cambio_estado = 0
            salto_solicitado = False
        
        frame_resized = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        annotated_frame = frame_resized.copy()
        color_gris_actual = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # --- A. CINEMÁTICA VERTICAL ---
        movimiento_y = 0
        if color_gris_previo is not None and puntos_previos is not None and len(puntos_previos) > 0:
            puntos_actuales, st, err = cv2.calcOpticalFlowPyrLK(color_gris_previo, color_gris_actual, puntos_previos, None, **lk_params)
            if puntos_actuales is not None:
                p_nuevos = puntos_actuales[st == 1]
                p_viejos = puntos_previos[st == 1]
                if len(p_nuevos) > 0:
                    desplazamientos_y = p_nuevos[:, 1] - p_viejos[:, 1]
                    movimiento_y = np.mean(desplazamientos_y)
                    historial_flujo_y.append(movimiento_y)
        
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 10 == 0 or puntos_previos is None:
            puntos_previos = cv2.goodFeaturesToTrack(color_gris_actual, mask=None, **feature_params)
        color_gris_previo = color_gris_actual.copy()

        tendencia_flujo_y = np.mean(historial_flujo_y) if len(historial_flujo_y) > 0 else 0
        
        # --- B. LECTURA DE IA ---
        results = model.predict(frame_resized, conf=0.15, imgsz=800, agnostic_nms=True, verbose=False)
        cajas_lineas = {"25yd": None, "lateral": None}
        objetos_detectados = []

        for box in results[0].boxes:
            nombre_clase = model.names[int(box.cls[0])].lower()
            xyxy = box.xyxy[0].cpu().numpy()
            cx, cy = obtener_centroide(xyxy)
            objetos_detectados.append({"clase": nombre_clase, "cy": cy})
            
            if "goal" in nombre_clase: cv2.circle(annotated_frame, (cx, cy), 8, (255, 0, 0), -1) 
            elif "cruce_t" in nombre_clase: cv2.circle(annotated_frame, (cx, cy), 8, (0, 0, 255), -1) 
            elif "25yd line" in nombre_clase: cajas_lineas["25yd"] = xyxy
            elif "lateral line" in nombre_clase: cajas_lineas["lateral"] = xyxy

        # --- C. GEOMETRÍA HÍBRIDA ---
        if cajas_lineas["25yd"] is not None and cajas_lineas["lateral"] is not None:
            recta_25 = extraer_linea_hough(frame_resized, cajas_lineas["25yd"])
            recta_lat = extraer_linea_hough(frame_resized, cajas_lineas["lateral"])
            nodo_virtual = calcular_interseccion(recta_25, recta_lat)
            if nodo_virtual is not None and (0 <= nodo_virtual[0] <= VIDEO_W) and (0 <= nodo_virtual[1] <= VIDEO_H):
                cv2.circle(annotated_frame, nodo_virtual, 10, (0, 0, 255), 2)

        # --- D. LÓGICA DE RECUPERACIÓN MEJORADA ---
        zona_actual_detectada = inferir_zona_horizonte(objetos_detectados)
        
        # La IA actualiza la memoria SOLO si está segura de la zona
        if zona_actual_detectada != "Zona_Transicion":
            ultima_zona_valida = zona_actual_detectada
            
        evento_trigger = None

        if tendencia_flujo_y > UMBRAL_FLUJO: direccion_camara = "Ataca_Arriba (Visita)"
        elif tendencia_flujo_y < -UMBRAL_FLUJO: direccion_camara = "Ataca_Abajo (Local)"
        else: direccion_camara = estado_posesion

        # Máquina de estados con confirmación en el tiempo
        if direccion_camara != estado_posesion and estado_posesion != "Indefinido":
            frames_cambio_estado += 1
            if frames_cambio_estado >= FRAMES_CONFIRMACION:
                # TURNOVER CONFIRMADO (Guardamos en la última zona que YOLO vio clara)
                metricas_recuperacion[ultima_zona_valida] += 1
                evento_trigger = f"RECUPERACION: {ultima_zona_valida}"
                
                registro_eventos.append({
                    "Frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "Minuto_Video": round(cap.get(cv2.CAP_PROP_POS_FRAMES) / fps_video / 60, 2),
                    "Nuevo_Estado": direccion_camara,
                    "Zona": ultima_zona_valida
                })
                estado_posesion = direccion_camara
                frames_cambio_estado = 0
        else:
            if direccion_camara == estado_posesion:
                frames_cambio_estado = 0 # Falsa alarma, reiniciamos el contador de seguridad

        # Caso inicial
        if estado_posesion == "Indefinido" and direccion_camara != "Indefinido":
            estado_posesion = direccion_camara

    # --- E. DIBUJAR INTERFAZ ---
    if annotated_frame is not None:
        display_frame = annotated_frame.copy()
        
        cv2.rectangle(display_frame, (0, 0), (VIDEO_W, 90), (15, 15, 15), -1)
        cv2.putText(display_frame, f"CINEMATICA: {estado_posesion}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Mostramos qué está viendo YOLO vs qué tenemos en memoria
        texto_zona = f"MEMORIA ZONA: {ultima_zona_valida}" if zona_actual_detectada == "Zona_Transicion" else f"VIENDO ZONA: {zona_actual_detectada}"
        color_zona = (0, 165, 255) if zona_actual_detectada == "Zona_Transicion" else (255, 255, 0)
        cv2.putText(display_frame, texto_zona, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_zona, 2)
        
        if 'evento_trigger' in locals() and evento_trigger:
            cv2.putText(display_frame, evento_trigger, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display_frame, "RECUPERACIONES ACUMULADAS", (VIDEO_W - 270, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset = 40
        for zona, conteo in metricas_recuperacion.items():
            cv2.putText(display_frame, f"{zona}: {conteo}", (VIDEO_W - 270, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 15

        cv2.rectangle(display_frame, (0, VIDEO_H - 50), (VIDEO_W, VIDEO_H), (40, 40, 40), -1)
        botones = [("<< 5s", 10, 80), ("< 1s", 90, 160), (estado_reproduccion, 170, 270), ("> 1s", 280, 350), (">> 5s", 360, 430)]
        for texto, x1, x2 in botones:
            color = (0, 150, 0) if texto == "PLAY" else (0, 0, 150) if texto == "PAUSE" else (100, 100, 100)
            cv2.rectangle(display_frame, (x1, VIDEO_H - 40), (x2, VIDEO_H - 10), color, -1)
            cv2.putText(display_frame, texto, (x1 + 10, VIDEO_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Dashboard Analitico V6.1", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord(' '): estado_reproduccion = "PAUSE" if estado_reproduccion == "PLAY" else "PLAY"

cap.release()
cv2.destroyAllWindows()

df_resultados = pd.DataFrame(registro_eventos)
df_resultados.to_csv("recuperaciones_hockey.csv", index=False)
print("\n--- PROCESAMIENTO FINALIZADO ---")
if not df_resultados.empty: print(df_resultados.head(10))