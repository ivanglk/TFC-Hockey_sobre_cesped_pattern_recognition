import cv2
import numpy as np
import pandas as pd
from collections import deque
from ultralytics import YOLO

print("--- INICIANDO SISTEMA V6.4: MOTOR SEMÁNTICO + DENSIDAD DE BOCHA + FILTRO DE CORDURA ---")

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
path_modelo = r"C:\Users\ivang\Desktop\Tesis_Hockey\models\best_v5.pt" 
path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\3raF-Inter_D_ 2024-VelezB_2-0SICb.mp4"

VIDEO_W, VIDEO_H = 800, 600

# ==========================================
# 2. VARIABLES DE ESTADO Y MÉTRICAS
# ==========================================
metricas_recuperacion = {"Z1_ArcoLocal_25yd": 0, "Z2_25yd_50yd_Local": 0, "Z3_50yd_25yd_Visita": 0, "Z4_25yd_ArcoVisita": 0}
historial_flujo_y = deque(maxlen=15) 
estado_posesion = "Indefinido" 
registro_eventos = [] 

ultima_zona_valida = "Z2_25yd_50yd_Local" 
frames_cambio_estado = 0
UMBRAL_FLUJO = 0.6  
FRAMES_CONFIRMACION = 5 

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
color_gris_previo = None; puntos_previos = None

# ==========================================
# 3. REPRODUCTOR DE VIDEO
# ==========================================
estado_reproduccion = "PLAY"
salto_solicitado = True 
cap = cv2.VideoCapture(path_video)
fps_video = cap.get(cv2.CAP_PROP_FPS) or 30

def click_botones(event, x, y, flags, param):
    global estado_reproduccion, salto_solicitado
    if event == cv2.EVENT_LBUTTONDOWN:
        f_actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if 550 <= y <= 590:
            if 10 <= x <= 80:    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f_actual - 5 * fps_video)); salto_solicitado = True
            elif 90 <= x <= 160: cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f_actual - 1 * fps_video)); salto_solicitado = True
            elif 170 <= x <= 270: estado_reproduccion = "PAUSE" if estado_reproduccion == "PLAY" else "PLAY"
            elif 280 <= x <= 350: cap.set(cv2.CAP_PROP_POS_FRAMES, f_actual + 1 * fps_video); salto_solicitado = True
            elif 360 <= x <= 430: cap.set(cv2.CAP_PROP_POS_FRAMES, f_actual + 5 * fps_video); salto_solicitado = True

# ==========================================
# 4. LÓGICA SEMÁNTICA Y DENSIDAD
# ==========================================
def obtener_metricas_caja(xyxy):
    x1, y1, x2, y2 = xyxy
    return (int((x1 + x2) / 2), int((y1 + y2) / 2), int(y2 - y1))

def inferir_zona_semantica(objetos_yolo, alto_pantalla, zona_previa):
    centro_accion = alto_pantalla / 2
    umbral_arco_grande = alto_pantalla * 0.08 
    
    for obj in objetos_yolo:
        if obj["clase"] == "goal":
            return "Z1_ArcoLocal_25yd" if obj["alto_box"] > umbral_arco_grande else "Z4_25yd_ArcoVisita"
                
    for obj in objetos_yolo:
        if obj["clase"] == "50yd line":
            return "Z2_25yd_50yd_Local" if obj["cy"] < centro_accion else "Z3_50yd_25yd_Visita"
            
    for obj in objetos_yolo:
        if obj["clase"] == "25yd line":
            if zona_previa in ["Z1_ArcoLocal_25yd", "Z2_25yd_50yd_Local"]:
                return "Z1_ArcoLocal_25yd" if obj["cy"] < centro_accion else "Z2_25yd_50yd_Local"
            else:
                return "Z3_50yd_25yd_Visita" if obj["cy"] > centro_accion else "Z4_25yd_ArcoVisita"
                
    return "Zona_Transicion"

def inferir_zona_disputa(jugadores_xy, radio_busqueda=120, min_jugadores=3):
    """Infiera la posición de la bocha basándose en la densidad de jugadores."""
    if len(jugadores_xy) < min_jugadores: return None, 0
    max_vecinos = 0
    centro_disputa = None
    for cx, cy in jugadores_xy:
        vecinos = [(nx, ny) for nx, ny in jugadores_xy if np.sqrt((cx - nx)**2 + (cy - ny)**2) < radio_busqueda]
        if len(vecinos) > max_vecinos:
            max_vecinos = len(vecinos)
            centro_disputa = (int(np.mean([v[0] for v in vecinos])), int(np.mean([v[1] for v in vecinos])))
    return (centro_disputa, max_vecinos) if max_vecinos >= min_jugadores else (None, 0)

# ==========================================
# 5. INICIALIZACIÓN Y ESCANEO
# ==========================================
cv2.namedWindow("Dashboard Analitico V6.4")
cv2.setMouseCallback("Dashboard Analitico V6.4", click_botones)
model = YOLO(path_modelo)

success_init, frame_init = cap.read()
if success_init:
    frame_calib = cv2.resize(frame_init, (VIDEO_W, VIDEO_H))
    res_init = model.predict(frame_calib, conf=0.15, imgsz=800, agnostic_nms=True, verbose=False)
    obj_init = []
    
    for box in res_init[0].boxes:
        confianza = float(box.conf[0])
        cls_name = model.names[int(box.cls[0])].lower()
        
        # FILTRO DE CORDURA ASIMÉTRICO: Exigimos más confianza a las líneas para evitar alucinaciones
        if "line" in cls_name and confianza < 0.35: continue
        
        cx, cy, alto = obtener_metricas_caja(box.xyxy[0].cpu().numpy())
        obj_init.append({"clase": cls_name, "cy": cy, "alto_box": alto})
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame_calib, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame_calib, f"{cls_name} ({confianza:.2f})", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    zona_detectada = inferir_zona_semantica(obj_init, VIDEO_H, "Z2_25yd_50yd_Local")
    
    cv2.rectangle(frame_calib, (0, int(VIDEO_H/2 - 40)), (VIDEO_W, int(VIDEO_H/2 + 60)), (20, 20, 20), -1)
    cv2.putText(frame_calib, "ESCANEO SEMANTICO INICIAL (CON FILTRO)", (150, int(VIDEO_H/2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame_calib, f"ZONA INFERIDA: {zona_detectada}", (180, int(VIDEO_H/2 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    while True:
        cv2.imshow("Dashboard Analitico V6.4", frame_calib)
        k = cv2.waitKey(1) & 0xFF
        if k == ord(' '): break
        elif k == ord('q'): exit()

cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
annotated_frame = None

# ==========================================
# 6. BUCLE PRINCIPAL
# ==========================================
while cap.isOpened():
    if estado_reproduccion == "PLAY" or salto_solicitado:
        success, frame = cap.read()
        if not success: break
        
        if salto_solicitado:
            color_gris_previo = None; puntos_previos = None
            historial_flujo_y.clear(); frames_cambio_estado = 0
            salto_solicitado = False
        
        frame_resized = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        annotated_frame = frame_resized.copy()
        color_gris_actual = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # --- A. CINEMÁTICA ---
        movimiento_y = 0
        if color_gris_previo is not None and puntos_previos is not None and len(puntos_previos) > 0:
            p_act, st, err = cv2.calcOpticalFlowPyrLK(color_gris_previo, color_gris_actual, puntos_previos, None, **lk_params)
            if p_act is not None:
                p_nuevos = p_act[st == 1]; p_viejos = puntos_previos[st == 1]
                if len(p_nuevos) > 0:
                    movimiento_y = np.mean(p_nuevos[:, 1] - p_viejos[:, 1])
                    historial_flujo_y.append(movimiento_y)
        
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 10 == 0 or puntos_previos is None:
            puntos_previos = cv2.goodFeaturesToTrack(color_gris_actual, mask=None, **feature_params)
        color_gris_previo = color_gris_actual.copy()
        tendencia_flujo_y = np.mean(historial_flujo_y) if len(historial_flujo_y) > 0 else 0
        
        # --- B. IA, FILTRO DE CORDURA Y JUGADORES ---
        results = model.predict(frame_resized, conf=0.15, imgsz=800, agnostic_nms=True, verbose=False)
        objetos_detectados = []
        coords_jugadores = []

        for box in results[0].boxes:
            confianza = float(box.conf[0])
            nombre_clase = model.names[int(box.cls[0])].lower()
            
            # Filtro Asimétrico
            if "line" in nombre_clase and confianza < 0.35: continue
            
            cx, cy, alto_box = obtener_metricas_caja(box.xyxy[0].cpu().numpy())
            objetos_detectados.append({"clase": nombre_clase, "cy": cy, "alto_box": alto_box})
            
            if "goal" in nombre_clase: cv2.circle(annotated_frame, (cx, cy), 8, (255, 0, 0), -1) 
            elif "25yd line" in nombre_clase: cv2.circle(annotated_frame, (cx, cy), 8, (0, 255, 0), -1)
            elif "player" in nombre_clase: coords_jugadores.append((cx, cy))

        # --- C. INFERENCIA DE BOCHA POR DENSIDAD ---
        centro_disputa, cant_implicados = inferir_zona_disputa(coords_jugadores, radio_busqueda=120, min_jugadores=4)
        if centro_disputa is not None:
            cv2.circle(annotated_frame, centro_disputa, 120, (0, 165, 255), 2)
            cv2.circle(annotated_frame, centro_disputa, 5, (0, 165, 255), -1) 
            cv2.putText(annotated_frame, f"DISPUTA: {cant_implicados} jug.", (centro_disputa[0] - 50, centro_disputa[1] - 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # --- D. MOTOR SEMÁNTICO Y REGISTRO ---
        zona_actual_detectada = inferir_zona_semantica(objetos_detectados, VIDEO_H, ultima_zona_valida)
        if zona_actual_detectada != "Zona_Transicion": ultima_zona_valida = zona_actual_detectada
            
        evento_trigger = None
        if tendencia_flujo_y > UMBRAL_FLUJO: direccion_camara = "Ataca_Arriba (Visita)"
        elif tendencia_flujo_y < -UMBRAL_FLUJO: direccion_camara = "Ataca_Abajo (Local)"
        else: direccion_camara = estado_posesion

        if direccion_camara != estado_posesion and estado_posesion != "Indefinido":
            frames_cambio_estado += 1
            if frames_cambio_estado >= FRAMES_CONFIRMACION:
                metricas_recuperacion[ultima_zona_valida] += 1
                evento_trigger = f"RECUPERACION: {ultima_zona_valida}"
                registro_eventos.append({
                    "Frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "Minuto_Video": round(cap.get(cv2.CAP_PROP_POS_FRAMES) / fps_video / 60, 2),
                    "Nuevo_Estado": direccion_camara,
                    "Zona": ultima_zona_valida
                })
                estado_posesion = direccion_camara; frames_cambio_estado = 0
        else:
            if direccion_camara == estado_posesion: frames_cambio_estado = 0 

        if estado_posesion == "Indefinido" and direccion_camara != "Indefinido": estado_posesion = direccion_camara

    # --- E. UI ---
    if annotated_frame is not None:
        display_frame = annotated_frame.copy()
        
        cv2.rectangle(display_frame, (0, 0), (VIDEO_W, 90), (15, 15, 15), -1)
        cv2.putText(display_frame, f"CINEMATICA: {estado_posesion}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
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

        cv2.imshow("Dashboard Analitico V6.4", display_frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    elif k == ord(' '): estado_reproduccion = "PAUSE" if estado_reproduccion == "PLAY" else "PLAY"

cap.release()
cv2.destroyAllWindows()

df_resultados = pd.DataFrame(registro_eventos)
df_resultados.to_csv("recuperaciones_hockey.csv", index=False)
print("\n--- PROCESAMIENTO FINALIZADO ---")
if not df_resultados.empty: print(df_resultados.head(10))