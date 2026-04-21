import cv2
import numpy as np
import pandas as pd
from collections import deque
from ultralytics import YOLO

print("--- INICIANDO SISTEMA V7.0: MASTER RELEASE (SEMÁNTICA + LEAKY BUCKET + PIZARRA 2D) ---")

# ==========================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================
path_modelo = r"C:\Users\ivang\Desktop\Tesis_Hockey\models\best_v5.pt" 
#path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\3raF-Inter_D_ 2024-VelezB_2-0SICb.mp4"
path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\6taFecha-2024-Vélez _B_ 2-0DAOM.mp4" 


VIDEO_W, VIDEO_H = 800, 600
PANEL_W = 250  # Ancho extra para la pizarra táctica 2D
TOTAL_W = VIDEO_W + PANEL_W

# ==========================================
# 2. VARIABLES DE ESTADO Y MÉTRICAS
# ==========================================
metricas_recuperacion = {"Z1_ArcoLocal_25yd": 0, "Z2_25yd_50yd_Local": 0, "Z3_50yd_25yd_Visita": 0, "Z4_25yd_ArcoVisita": 0}
registro_eventos = [] 

estado_posesion = "Indefinido" 
ultima_zona_valida = "Z2_25yd_50yd_Local" 

# --- SISTEMA DE EVIDENCIA (LEAKY BUCKET) ---
evidencia_posesion = 0      
UMBRAL_MOVIMIENTO = 0.3     
UMBRAL_EVIDENCIA_DINAMICO = 8 # Más sensible a contraataques rápidos

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
color_gris_previo = None; puntos_previos = None

# ==========================================
# 3. REPRODUCTOR DE VIDEO Y CONTROLES
# ==========================================
estado_reproduccion = "PLAY"
salto_solicitado = True 
cap = cv2.VideoCapture(path_video)
fps_video = cap.get(cv2.CAP_PROP_FPS) or 30

def click_botones(event, x, y, flags, param):
    global estado_reproduccion, salto_solicitado
    if event == cv2.EVENT_LBUTTONDOWN:
        f_actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Los botones están dibujados en la parte inferior izquierda (0 a 430 px)
        if (VIDEO_H - 50) <= y <= VIDEO_H:
            if 10 <= x <= 80:    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f_actual - 5 * fps_video)); salto_solicitado = True
            elif 90 <= x <= 160: cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f_actual - 1 * fps_video)); salto_solicitado = True
            elif 170 <= x <= 270: estado_reproduccion = "PAUSE" if estado_reproduccion == "PLAY" else "PLAY"
            elif 280 <= x <= 350: cap.set(cv2.CAP_PROP_POS_FRAMES, f_actual + 1 * fps_video); salto_solicitado = True
            elif 360 <= x <= 430: cap.set(cv2.CAP_PROP_POS_FRAMES, f_actual + 5 * fps_video); salto_solicitado = True

# ==========================================
# 4. LÓGICA SEMÁNTICA "A PRUEBA DE ZOOM"
# ==========================================
def obtener_metricas_caja(xyxy):
    x1, y1, x2, y2 = xyxy
    return (int((x1 + x2) / 2), int((y1 + y2) / 2), int(y2 - y1))

def inferir_zona_semantica(objetos_yolo, alto_pantalla, zona_previa):
    centro_accion = alto_pantalla / 2
    
    # 1. Arcos: Inmune al Zoom. Si está abajo en la pantalla es Local, si está arriba es Visita.
    for obj in objetos_yolo:
        if obj["clase"] == "goal":
            return "Z1_ArcoLocal_25yd" if obj["cy"] > centro_accion else "Z4_25yd_ArcoVisita"
                
    # 2. Línea 50yd
    for obj in objetos_yolo:
        if obj["clase"] == "50yd line":
            return "Z2_25yd_50yd_Local" if obj["cy"] < centro_accion else "Z3_50yd_25yd_Visita"
            
    # 3. Línea 25yd dependiente de la memoria
    for obj in objetos_yolo:
        if obj["clase"] == "25yd line":
            if zona_previa in ["Z1_ArcoLocal_25yd", "Z2_25yd_50yd_Local"]:
                return "Z1_ArcoLocal_25yd" if obj["cy"] < centro_accion else "Z2_25yd_50yd_Local"
            else:
                return "Z3_50yd_25yd_Visita" if obj["cy"] > centro_accion else "Z4_25yd_ArcoVisita"
                
    return "Zona_Transicion"

def inferir_zona_disputa(jugadores_xy, radio_busqueda=120, min_jugadores=4):
    if len(jugadores_xy) < min_jugadores: return None, 0
    max_vecinos = 0; centro_disputa = None
    for cx, cy in jugadores_xy:
        vecinos = [(nx, ny) for nx, ny in jugadores_xy if np.sqrt((cx - nx)**2 + (cy - ny)**2) < radio_busqueda]
        if len(vecinos) > max_vecinos:
            max_vecinos = len(vecinos)
            centro_disputa = (int(np.mean([v[0] for v in vecinos])), int(np.mean([v[1] for v in vecinos])))
    return (centro_disputa, max_vecinos) if max_vecinos >= min_jugadores else (None, 0)

# ==========================================
# 5. PIZARRA TÁCTICA 2D (MINIMAPA)
# ==========================================
def dibujar_minimapa(lienzo, jugadores_xy, centro_disputa, zona_actual):
    """Proyecta proporcionalmente los jugadores detectados en un minimapa 2D."""
    # Fondo verde para el minimapa
    cv2.rectangle(lienzo, (VIDEO_W, 0), (TOTAL_W, VIDEO_H), (34, 100, 34), -1)
    
    # Dibujar líneas del campo 2D
    cv2.line(lienzo, (VIDEO_W + 20, int(VIDEO_H * 0.1)), (TOTAL_W - 20, int(VIDEO_H * 0.1)), (255, 255, 255), 2) # Fondo Visita
    cv2.line(lienzo, (VIDEO_W + 20, int(VIDEO_H * 0.3)), (TOTAL_W - 20, int(VIDEO_H * 0.3)), (200, 200, 200), 1) # 25yd Visita
    cv2.line(lienzo, (VIDEO_W + 20, int(VIDEO_H * 0.5)), (TOTAL_W - 20, int(VIDEO_H * 0.5)), (255, 255, 255), 2) # 50yd
    cv2.line(lienzo, (VIDEO_W + 20, int(VIDEO_H * 0.7)), (TOTAL_W - 20, int(VIDEO_H * 0.7)), (200, 200, 200), 1) # 25yd Local
    cv2.line(lienzo, (VIDEO_W + 20, int(VIDEO_H * 0.9)), (TOTAL_W - 20, int(VIDEO_H * 0.9)), (255, 255, 255), 2) # Fondo Local
    
    cv2.putText(lienzo, "PIZARRA TACTICA 2D", (VIDEO_W + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(lienzo, f"Zona: {zona_actual[:2]}", (VIDEO_W + 15, VIDEO_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Proyectar Jugadores (Simulación Proporcional Pseudo-Homográfica)
    for (x, y) in jugadores_xy:
        map_x = VIDEO_W + 20 + int((x / VIDEO_W) * (PANEL_W - 40))
        map_y = int((y / VIDEO_H) * (VIDEO_H * 0.8)) + int(VIDEO_H * 0.1)
        cv2.circle(lienzo, (map_x, map_y), 4, (0, 255, 255), -1) # Jugadores Amarillos
        
    # Proyectar Centro de Gravedad (Bocha inferida)
    if centro_disputa is not None:
        dx, dy = centro_disputa
        map_dx = VIDEO_W + 20 + int((dx / VIDEO_W) * (PANEL_W - 40))
        map_dy = int((dy / VIDEO_H) * (VIDEO_H * 0.8)) + int(VIDEO_H * 0.1)
        cv2.circle(lienzo, (map_dx, map_dy), 8, (0, 165, 255), -1) # Foco de presión Naranja
        cv2.circle(lienzo, (map_dx, map_dy), 15, (0, 165, 255), 1)

# ==========================================
# 6. INICIALIZACIÓN DE VENTANA
# ==========================================
cv2.namedWindow("Dashboard Analitico V7.0")
cv2.setMouseCallback("Dashboard Analitico V7.0", click_botones)
model = YOLO(path_modelo)

# ==========================================
# 7. BUCLE PRINCIPAL
# ==========================================
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 

while cap.isOpened():
    if estado_reproduccion == "PLAY" or salto_solicitado:
        success, frame = cap.read()
        if not success: break
        
        if salto_solicitado:
            color_gris_previo = None; puntos_previos = None
            evidencia_posesion = 0 # Vaciamos el balde al saltar en el video
            salto_solicitado = False
        
        frame_resized = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        annotated_frame = frame_resized.copy()
        color_gris_actual = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # Lienzo unificado (Video + Minimapa)
        display_frame = np.zeros((VIDEO_H, TOTAL_W, 3), dtype=np.uint8)
        
        # --- A. CINEMÁTICA Y LEAKY BUCKET ---
        movimiento_y = 0
        if color_gris_previo is not None and puntos_previos is not None and len(puntos_previos) > 0:
            p_act, st, err = cv2.calcOpticalFlowPyrLK(color_gris_previo, color_gris_actual, puntos_previos, None, **lk_params)
            if p_act is not None:
                p_nuevos = p_act[st == 1]; p_viejos = puntos_previos[st == 1]
                if len(p_nuevos) > 0:
                    movimiento_y = np.mean(p_nuevos[:, 1] - p_viejos[:, 1])
                    
                    # LLENADO DEL BALDE
                    if movimiento_y > UMBRAL_MOVIMIENTO:
                        evidencia_posesion += 1
                    elif movimiento_y < -UMBRAL_MOVIMIENTO:
                        evidencia_posesion -= 1
                    else:
                        # FUGA (El balde gotea si la cámara se detiene)
                        if evidencia_posesion > 0: evidencia_posesion -= 0.5
                        elif evidencia_posesion < 0: evidencia_posesion += 0.5
                        
                    evidencia_posesion = max(-12, min(12, evidencia_posesion))
        
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 5 == 0 or puntos_previos is None or len(puntos_previos) < 10:
            puntos_previos = cv2.goodFeaturesToTrack(color_gris_actual, mask=None, **feature_params)
        color_gris_previo = color_gris_actual.copy()
        
        # --- B. IA, EXCLUSIÓN ESPACIAL Y COORDENADAS ---
        results = model.predict(frame_resized, conf=0.15, imgsz=800, agnostic_nms=True, verbose=False)
        objetos_detectados = []
        coords_jugadores = []

        # 1. Encontrar Arcos para Escudo Geométrico
        cajas_arcos_loop = [box.xyxy[0].cpu().numpy() for box in results[0].boxes if model.names[int(box.cls[0])].lower() == "goal"]

        # 2. Filtrado y extracción
        for box in results[0].boxes:
            nombre_clase = model.names[int(box.cls[0])].lower()
            xyxy = box.xyxy[0].cpu().numpy()
            cx, cy, alto_box = obtener_metricas_caja(xyxy)
            
            # Escudo: Ignorar líneas "dentro" de los arcos
            if "line" in nombre_clase:
                adentro_arco = any((a[0]-15) < cx < (a[2]+15) and (a[1]-15) < cy < (a[3]+15) for a in cajas_arcos_loop)
                if adentro_arco: continue
            
            objetos_detectados.append({"clase": nombre_clase, "cy": cy, "alto_box": alto_box})
            
            if "goal" in nombre_clase: cv2.circle(annotated_frame, (cx, cy), 8, (255, 0, 0), -1) 
            elif "25yd line" in nombre_clase: cv2.circle(annotated_frame, (cx, cy), 8, (0, 255, 0), -1)
            elif "player" in nombre_clase: coords_jugadores.append((cx, cy))

        # --- C. INFERENCIA DE BOCHA POR DENSIDAD ---
        centro_disputa, cant_implicados = inferir_zona_disputa(coords_jugadores, radio_busqueda=120, min_jugadores=4)
        if centro_disputa is not None:
            cv2.circle(annotated_frame, centro_disputa, 120, (0, 165, 255), 2)
            cv2.circle(annotated_frame, centro_disputa, 5, (0, 165, 255), -1) 

        # --- D. MOTOR SEMÁNTICO Y REGISTRO ---
        zona_actual_detectada = inferir_zona_semantica(objetos_detectados, VIDEO_H, ultima_zona_valida)
        if zona_actual_detectada != "Zona_Transicion": ultima_zona_valida = zona_actual_detectada
            
        evento_trigger = None
        nuevo_estado = estado_posesion

        # Evaluamos el Balde de Evidencia
        if evidencia_posesion >= UMBRAL_EVIDENCIA_DINAMICO:
            nuevo_estado = "Ataca_Arriba (Visita)"
        elif evidencia_posesion <= -UMBRAL_EVIDENCIA_DINAMICO:
            nuevo_estado = "Ataca_Abajo (Local)"

        # ¡RESETEO INSTANTÁNEO TRAS CONFIRMAR TURNOVER!
        if nuevo_estado != estado_posesion:
            if estado_posesion != "Indefinido": 
                metricas_recuperacion[ultima_zona_valida] += 1
                evento_trigger = f"RECUPERACION EN: {ultima_zona_valida}"
                registro_eventos.append({
                    "Frame": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    "Minuto_Video": round(cap.get(cv2.CAP_PROP_POS_FRAMES) / fps_video / 60, 2),
                    "Nuevo_Estado": nuevo_estado,
                    "Zona_Recuperacion": ultima_zona_valida
                })
            estado_posesion = nuevo_estado
            evidencia_posesion = 0 # Vaciamos el balde para poder captar el siguiente contragolpe rápido

        # --- E. RENDERIZADO DE INTERFAZ (UI) ---
        # 1. Pegar el video en el lienzo principal
        display_frame[0:VIDEO_H, 0:VIDEO_W] = annotated_frame
        
        # 2. Dibujar Pizarra Táctica 2D
        dibujar_minimapa(display_frame, coords_jugadores, centro_disputa, ultima_zona_valida)

        # 3. Panel Analítico Superior
        cv2.rectangle(display_frame, (0, 0), (VIDEO_W, 90), (15, 15, 15), -1)
        cv2.putText(display_frame, f"CINEMATICA: {estado_posesion} (Evid: {evidencia_posesion:.1f})", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
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

        # 4. Botonera Inferior
        cv2.rectangle(display_frame, (0, VIDEO_H - 50), (VIDEO_W, VIDEO_H), (40, 40, 40), -1)
        botones = [("<< 5s", 10, 80), ("< 1s", 90, 160), (estado_reproduccion, 170, 270), ("> 1s", 280, 350), (">> 5s", 360, 430)]
        for texto, x1, x2 in botones:
            color = (0, 150, 0) if texto == "PLAY" else (0, 0, 150) if texto == "PAUSE" else (100, 100, 100)
            cv2.rectangle(display_frame, (x1, VIDEO_H - 40), (x2, VIDEO_H - 10), color, -1)
            cv2.putText(display_frame, texto, (x1 + 10, VIDEO_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Dashboard Analitico V7.0", display_frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    elif k == ord(' '): estado_reproduccion = "PAUSE" if estado_reproduccion == "PLAY" else "PLAY"

cap.release()
cv2.destroyAllWindows()

# ==========================================
# 8. EXPORTACIÓN DE RESULTADOS
# ==========================================
df_resultados = pd.DataFrame(registro_eventos)
df_resultados.to_csv("recuperaciones_hockey_final.csv", index=False)
print("\n--- PROCESAMIENTO FINALIZADO ---")
if not df_resultados.empty: 
    print("Archivo 'recuperaciones_hockey_final.csv' guardado exitosamente.")
    print(df_resultados.head(10))
else:
    print("No se registraron cambios de posesión en la ventana de tiempo evaluada.")