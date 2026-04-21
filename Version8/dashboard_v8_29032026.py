import cv2
import numpy as np
import pandas as pd
from collections import deque
from ultralytics import YOLO

print("--- INICIANDO SISTEMA V7.1: MASTER RELEASE (SEMÁNTICA + LEAKY BUCKET + PIZARRA DE HOCKEY PROFESIONAL + EQUIPOS) ---")

# ==========================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================
# ⚠️ Actualiza estas rutas a las de tu PC
path_modelo = r"C:\Users\ivang\Desktop\Tesis_Hockey\models\best_v5.pt" 
#path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\3raF-Inter_D_ 2024-VelezB_2-0SICb.mp4"
path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\6taFecha-2024-Vélez _B_ 2-0DAOM.mp4" 


VIDEO_W, VIDEO_H = 800, 600
PANEL_W = 280  # Ancho extra para la pizarra táctica 2D reglamentaria
TOTAL_W = VIDEO_W + PANEL_W

# ==========================================
# 2. VARIABLES DE ESTADO Y MÉTRICAS
# ==========================================
metricas_recuperacion = {"Z1_ArcoLocal_25yd": 0, "Z2_25yd_50yd_Local": 0, "Z3_50yd_25yd_Visita": 0, "Z4_25yd_ArcoVisita": 0}
registro_eventos = [] 

estado_posesion = "Indefinido" 
ultima_zona_valida = "Z2_25yd_50yd_Local" 

# --- CONTROL DE INTERFAZ (BOTONES) ---
mostrar_anotaciones_ingeniero = True # Alternar entre Ingeniero y DT

# --- SISTEMA DE EVIDENCIA (LEAKY BUCKET) - SUPER SENSIBLE ---
evidencia_posesion = 0      
UMBRAL_MOVIMIENTO = 0.05     # ⚠️ CORRECCIÓN: Súper sensibilidad (de 0.3 a 0.05) para captar paneos suaves
UMBRAL_EVIDENCIA_DINAMICO = 6 # Llenado rápido del balde

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
    global estado_reproduccion, salto_solicitado, mostrar_anotaciones_ingeniero
    if event == cv2.EVENT_LBUTTONDOWN:
        f_actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
        # Los botones están dibujados en la parte inferior izquierda (0 a 590 px)
        if (VIDEO_H - 50) <= y <= VIDEO_H:
            if 10 <= x <= 80:    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f_actual - 5 * fps_video)); salto_solicitado = True
            elif 90 <= x <= 160: cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f_actual - 1 * fps_video)); salto_solicitado = True
            elif 170 <= x <= 270: estado_reproduccion = "PAUSE" if estado_reproduccion == "PLAY" else "PLAY"
            elif 280 <= x <= 350: cap.set(cv2.CAP_PROP_POS_FRAMES, f_actual + 1 * fps_video); salto_solicitado = True
            elif 360 <= x <= 430: cap.set(cv2.CAP_PROP_POS_FRAMES, f_actual + 5 * fps_video); salto_solicitado = True
            # Nuevo Botón de Alternar Anotaciones (ON/OFF)
            elif 440 <= x <= 590: mostrar_anotaciones_ingeniero = not mostrar_anotaciones_ingeniero

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
    for (cx, cy, color) in jugadores_xy:
        vecinos = [(nx, ny) for (nx, ny, ncol) in jugadores_xy if np.sqrt((cx - nx)**2 + (cy - ny)**2) < radio_busqueda]
        if len(vecinos) > max_vecinos:
            max_vecinos = len(vecinos)
            centro_disputa = (int(np.mean([v[0] for v in vecinos])), int(np.mean([v[1] for v in vecinos])))
    return (centro_disputa, max_vecinos) if max_vecinos >= min_jugadores else (None, 0)

# ==========================================
# 5. PIZARRA TÁCTICA 2D REGLAMENTARIA DE HOCKEY
# ==========================================
def dibujar_minimapa(lienzo, jugadores_info, centro_disputa, zona_actual):
    """Dibuja un campo reglamentario de hockey 2D y proyecta los equipos."""
    field_x1, field_y1 = VIDEO_W + 20, int(VIDEO_H * 0.05)
    field_w, field_h = PANEL_W - 40, int(VIDEO_H * 0.9)
    field_x2, field_y2 = field_x1 + field_w, field_y1 + field_h

    # Fondo verde reglamentario para el minimapa
    cv2.rectangle(lienzo, (VIDEO_W, 0), (TOTAL_W, VIDEO_H), (45, 115, 45), -1)
    
    # --- DIBUJO DEL CAMPO REGLAMENTARIO DE HOCKEY ---
    field_border_color = (255, 255, 255)
    
    # Líneas perimetrales (Fondo y Laterales)
    cv2.rectangle(lienzo, (field_x1, field_y1), (field_x2, field_y2), field_border_color, 2)
    
    # Línea central (50yd)
    cy_y = field_y1 + int(field_h * 0.5)
    cv2.line(lienzo, (field_x1, cy_y), (field_x2, cy_y), field_border_color, 2)
    cv2.circle(lienzo, (field_x1 + int(field_w*0.5), cy_y), int(field_w * 0.05), field_border_color, 1) # Punto central simbólico

    # Líneas de 23 metros (25yd esquemáticas)
    cv2.line(lienzo, (field_x1, field_y1 + int(field_h * 0.25)), (field_x2, field_y1 + int(field_h * 0.25)), (210, 210, 210), 1)
    cv2.line(lienzo, (field_x1, field_y1 + int(field_h * 0.75)), (field_x2, field_y1 + int(field_h * 0.75)), (210, 210, 210), 1)

    # Áreas de Gol esquemáticas (Círculo reglamentario comprimido en perspectiva cónica)
    goal_circle_color = (250, 250, 250)
    goal_w = int(field_w * 0.6)
    goal_h = int(field_h * 0.1)
    
    # Área Visita (Z4)
    gx1_vis = field_x1 + int(field_w*0.5 - goal_w*0.5)
    cv2.ellipse(lienzo, (field_x1 + int(field_w*0.5), field_y1), (goal_w // 2, goal_h), 0, 0, 180, goal_circle_color, 1) # Semicírculo
    cv2.rectangle(lienzo, (gx1_vis, field_y1), (gx1_vis + goal_w, field_y1 - 10), field_border_color, -1) # Arco
    
    # Área Local (Z1)
    cv2.ellipse(lienzo, (field_x1 + int(field_w*0.5), field_y2), (goal_w // 2, int(goal_h*1.5)), 0, 180, 360, goal_circle_color, 1)
    cv2.rectangle(lienzo, (gx1_vis, field_y2), (gx1_vis + goal_w, field_y2 + 10), field_border_color, -1)

    # --- UI Y TEXTOS ---
    cv2.putText(lienzo, "PIZARRA TACTICA 2D", (VIDEO_W + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Icono de Cámara (Perspectiva Cabecera) en Z4
    cam_x, cam_y = field_x2 - 15, field_y1 + 15
    cv2.rectangle(lienzo, (cam_x - 10, cam_y - 5), (cam_x + 10, cam_y + 5), (200, 200, 200), 1) # Esquemático
    cv2.rectangle(lienzo, (cam_x + 2, cam_y - 8), (cam_x + 8, cam_y + 8), (200, 200, 200), -1) 
    cv2.putText(lienzo, "HighBehind", (field_x2 - 80, field_y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Etiquetas de Zonas para el Técnico
    color_z4, color_z3 = (0, 0, 255) if zona_actual == "Z4_25yd_ArcoVisita" else (255, 255, 255), (0, 165, 255) if zona_actual == "Z3_50yd_25yd_Visita" else (255, 255, 255)
    color_z2, color_z1 = (0, 255, 255) if zona_actual == "Z2_25yd_50yd_Local" else (255, 255, 255), (0, 255, 0) if zona_actual == "Z1_ArcoLocal_25yd" else (255, 255, 255)
    cv2.putText(lienzo, "Z4", (field_x1 - 18, field_y1 + int(field_h * 0.12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_z4, 2)
    cv2.putText(lienzo, "Z3", (field_x1 - 18, field_y1 + int(field_h * 0.37)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_z3, 2)
    cv2.putText(lienzo, "Z2", (field_x1 - 18, field_y1 + int(field_h * 0.62)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_z2, 2)
    cv2.putText(lienzo, "Z1", (field_x1 - 18, field_y1 + int(field_h * 0.87)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_z1, 2)

    # Leyenda de Equipos
    cv2.circle(lienzo, (field_x1, VIDEO_H - 15), 4, (255, 255, 255), -1) # Claro
    cv2.putText(lienzo, "Claro", (field_x1 + 10, VIDEO_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.circle(lienzo, (field_x1 + 70, VIDEO_H - 15), 4, (0, 0, 0), -1) # Oscuro
    cv2.putText(lienzo, "Oscuro", (field_x1 + 80, VIDEO_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # --- PROYECCIÓN DE JUGADORES (DISTINCIÓN DE EQUIPOS) ---
    for (x, y, team_color) in jugadores_info:
        # Transformación Pseudo-Homográfica Proporcional
        map_x = field_x1 + int((x / VIDEO_W) * field_w)
        # Comprimimos la perspectiva hacia el fondo esquemáticamente
        depth_scale = 0.8 + (0.2 * (y / VIDEO_H)) # 0.8 arriba, 1.0 abajo
        comprimed_y = y * depth_scale
        map_y = field_y1 + int((comprimed_y / VIDEO_H) * field_h)
        
        # Dibujar punto en el minimapa con el color del equipo inferido
        cv2.circle(lienzo, (map_x, map_y), 4, team_color, -1) 
        
    # --- PROYECCIÓN DE FOCO DE DISPUTA ---
    if centro_disputa is not None:
        dx, dy = centro_disputa
        map_dx = field_x1 + int((dx / VIDEO_W) * field_w)
        d_scale = 0.8 + (0.2 * (dy / VIDEO_H))
        comprimed_dy = dy * d_scale
        map_dy = field_y1 + int((comprimed_dy / VIDEO_H) * field_h)
        
        # Aura Naranja simbólica
        cv2.circle(lienzo, (map_dx, map_dy), 12, (0, 165, 255), 1) # Contorno
        cv2.circle(lienzo, (map_dx, map_dy), 5, (0, 165, 255), -1) # Punto central

# ==========================================
# 6. INICIALIZACIÓN DE VENTANA
# ==========================================
cv2.namedWindow("Dashboard Analitico V7.1")
cv2.setMouseCallback("Dashboard Analitico V7.1", click_botones)
model = YOLO(path_modelo)

# ==========================================
# 7. BUCLE PRINCIPAL
# ==========================================
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
annotated_frame = None

while cap.isOpened():
    if estado_reproduccion == "PLAY" or salto_solicitado:
        success, frame = cap.read()
        if not success: break
        
        if salto_solicitado:
            color_gris_previo = None; puntos_previos = None
            evidencia_posesion = 0 # Vaciamos el balde al saltar en el video
            salto_solicitado = False
        
        frame_resized = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        
        # annotated_frame es la imagen que usará el ingeniero (con depuración)
        # frame_resized es la imagen limpia que usará el DT
        annotated_frame = frame_resized.copy()
        
        color_gris_actual = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        
        # Lienzo unificado (Video + Minimapa)
        display_frame = np.zeros((VIDEO_H, TOTAL_W, 3), dtype=np.uint8)
        
        # --- A. CINEMÁTICA Y LEAKY BUCKET (SÚPER SENSIBLE) ---
        movimiento_y = 0
        if color_gris_previo is not None and puntos_previos is not None and len(puntos_previos) > 0:
            p_act, st, err = cv2.calcOpticalFlowPyrLK(color_gris_previo, color_gris_actual, puntos_previos, None, **lk_params)
            if p_act is not None:
                p_nuevos = p_act[st == 1]; p_viejos = puntos_previos[st == 1]
                if len(p_nuevos) > 0:
                    movimiento_y = np.mean(p_nuevos[:, 1] - p_viejos[:, 1])
                    
                    # LLENADO DEL BALDE (Cualquier movimiento mínimo suma)
                    if movimiento_y > UMBRAL_MOVIMIENTO:
                        evidencia_posesion += 1
                    elif movimiento_y < -UMBRAL_MOVIMIENTO:
                        evidencia_posesion -= 1
                    else:
                        # FUGA (El balde gotea rápido si la cámara se detiene)
                        if evidencia_posesion > 0: evidencia_posesion -= 1.0
                        elif evidencia_posesion < 0: evidencia_posesion += 1.0
                        
                    evidencia_posesion = max(-10, min(10, evidencia_posesion))
        
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 5 == 0 or puntos_previos is None or len(puntos_previos) < 10:
            puntos_previos = cv2.goodFeaturesToTrack(color_gris_actual, mask=None, **feature_params)
        color_gris_previo = color_gris_actual.copy()
        
        # --- B. IA, EXCLUSIÓN ESPACIAL Y COORDENADAS (DISTINCIÓN DE EQUIPOS INFERIDA) ---
        results = model.predict(frame_resized, conf=0.15, imgsz=800, agnostic_nms=True, verbose=False)
        objetos_detectados = []
        coords_jugadores_minimapa = [] # Guardaremos (x, y, team_color)

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
            
            # Dibujos de depuración SÓLO para Ingeniero
            if mostrar_anotaciones_ingeniero:
                if "goal" in nombre_clase: cv2.circle(annotated_frame, (cx, cy), 8, (255, 0, 0), -1) 
                elif "25yd line" in nombre_clase: cv2.circle(annotated_frame, (cx, cy), 8, (0, 255, 0), -1)

            # Lógica de Distinción de Equipos
            if "player" in nombre_clase:
                # Segmentación de Color HSV dentro de la caja de la jugadora
                x1, y1, x2, y2 = map(int, xyxy)
                # Asegurar límites válidos
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(VIDEO_W, x2), min(VIDEO_H, y2)
                if x2 > x1 and y2 > y1:
                    player_crop = frame_resized[y1:y2, x1:x2]
                    player_hsv = cv2.cvtColor(player_crop, cv2.COLOR_BGR2HSV)
                    # Calculamos el brillo promedio del uniforme (Canal V)
                    mean_val = np.mean(player_hsv[:, :, 2])
                    
                    # Umbral Empírico: Si brilla más de 150 es Claro, si no Oscuro
                    if mean_val > 150: 
                        team_color = (255, 255, 255) # Blanco para Claro
                    else: 
                        team_color = (0, 0, 0) # Negro para Oscuro
                    
                    coords_jugadores_minimapa.append((cx, cy, team_color))
                    
                    # Dibuja puntito amarillo en video principal SÓLO para Ingeniero
                    if mostrar_anotaciones_ingeniero:
                        cv2.circle(annotated_frame, (cx, cy), 3, (0, 255, 255), -1) 

        # --- C. INFERENCIA DE BOCHA POR DENSIDAD ---
        centro_disputa, cant_implicados = inferir_zona_disputa(coords_jugadores_minimapa, radio_busqueda=120, min_jugadores=4)
        
        # Dibuja Aura Naranja simbólica SÓLO si Ingeniero está activado
        if mostrar_anotaciones_ingeniero and centro_disputa is not None:
            cv2.circle(annotated_frame, centro_disputa, 120, (0, 165, 255), 2)
            cv2.circle(annotated_frame, centro_disputa, 5, (0, 165, 255), -1) 

        # --- D. MOTOR SEMÁNTICO Y REGISTRO ---
        zona_actual_detectada = inferir_zona_semantica(objetos_detectados, VIDEO_H, ultima_zona_valida)
        if zona_actual_detectada != "Zona_Transicion": ultima_zona_valida = zona_actual_detectada
            
        evento_trigger = None
        nuevo_estado = estado_posesion

        # Evaluamos el Balde de Evidencia (con umbral más bajo=6 para reaccionar rápido)
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
        # 1. Pegar el video (Limpio o Anotado) en el lienzo principal
        if mostrar_anotaciones_ingeniero:
            display_frame[0:VIDEO_H, 0:VIDEO_W] = annotated_frame
        else:
            display_frame[0:VIDEO_H, 0:VIDEO_W] = frame_resized
        
        # 2. Dibujar Pizarra Táctica 2D REGLAMENTARIA
        dibujar_minimapa(display_frame, coords_jugadores_minimapa, centro_disputa, ultima_zona_valida)

        # 3. Panel Analítico Superior (Siempre visible)
        cv2.rectangle(display_frame, (0, 0), (VIDEO_W, 90), (15, 15, 15), -1)
        # Mostramos la evidencia con 1 decimal para ver cómo se llena (0.0 a 10.0)
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
        
        # Lista de botones [Texto, x1, x2]
        botones = [("<< 5s", 10, 80), ("< 1s", 90, 160), (estado_reproduccion, 170, 270), ("> 1s", 280, 350), (">> 5s", 360, 430)]
        for texto, x1, x2 in botones:
            color = (0, 150, 0) if texto == "PLAY" else (0, 0, 150) if texto == "PAUSE" else (100, 100, 100)
            cv2.rectangle(display_frame, (x1, VIDEO_H - 40), (x2, VIDEO_H - 10), color, -1)
            cv2.putText(display_frame, texto, (x1 + 10, VIDEO_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        # Nuevo Botón: ANOTACIONES (Alternar Ingeniero vs DT)
        bx1, bx2 = 440, 590
        b_color = (130, 130, 30) if mostrar_anotaciones_ingeniero else (60, 60, 60)
        b_text = "ANOTAC (ON)" if mostrar_anotaciones_ingeniero else "ANOTAC (OFF)"
        cv2.rectangle(display_frame, (bx1, VIDEO_H - 40), (bx2, VIDEO_H - 10), b_color, -1)
        cv2.putText(display_frame, b_text, (bx1 + 5, VIDEO_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Dashboard Analitico V7.1", display_frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    elif k == ord(' '): estado_reproduccion = "PAUSE" if estado_reproduccion == "PLAY" else "PLAY"

cap.release()
cv2.destroyAllWindows()

# ==========================================
# 8. EXPORTACIÓN DE RESULTADOS
# ==========================================
import os

# Forzamos la ruta absoluta sin importar desde dónde se ejecute VS Code
ruta_salida = r"C:\Users\ivang\Desktop\Tesis_Hockey\recuperaciones_hockey_final.csv"

df_resultados = pd.DataFrame(registro_eventos)

# Guardamos el archivo
df_resultados.to_csv(ruta_salida, index=False)

print("\n" + "="*50)
print("--- PROCESAMIENTO FINALIZADO CORRECTAMENTE ---")
print("="*50)

if not df_resultados.empty: 
    print(f"¡ÉXITO! Se registraron {len(df_resultados)} recuperaciones.")
    print(f"EL ARCHIVO FUE CREADO EXACTAMENTE AQUÍ:\n>>> {ruta_salida} <<<")
    print("-" * 50)
    print(df_resultados.head(10))
else:
    print("ATENCIÓN: El video terminó o se presionó 'q', pero NO SE REGISTRARON CAMBIOS DE POSESIÓN.")
    print("El archivo se creó, pero la tabla está vacía.")