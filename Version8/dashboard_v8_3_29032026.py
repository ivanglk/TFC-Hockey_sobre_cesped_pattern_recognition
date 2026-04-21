#==================================
# Esta es la versión más fiable hasta el momento. Modificada con Tabla Táctica Agrupada.
# VERSION FINAL !!!! :) 
#==================================

import cv2
import numpy as np
import pandas as pd
from collections import deque
from ultralytics import YOLO

print("--- INICIANDO SISTEMA V8.1: CRONÓMETRO MM:SS + TABLA TÁCTICA AGRUPADA ---")

# ==========================================
# 1. CONFIGURACIÓN GENERAL
# ==========================================
path_modelo = r"C:\Users\ivang\Desktop\Tesis_Hockey\models\best_v5.pt" 
path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\6taFecha-2024-Vélez _B_ 2-0DAOM.mp4" 

VIDEO_W, VIDEO_H = 800, 600
PANEL_W = 280  
TOTAL_W = VIDEO_W + PANEL_W

# ==========================================
# 2. VARIABLES DE ESTADO Y MÉTRICAS POR EQUIPO
# ==========================================
metricas_recuperacion = {
    "Local": {"Z1_ArcoLocal_25yd": 0, "Z2_25yd_50yd_Local": 0, "Z3_50yd_25yd_Visita": 0, "Z4_25yd_ArcoVisita": 0},
    "Visita": {"Z1_ArcoLocal_25yd": 0, "Z2_25yd_50yd_Local": 0, "Z3_50yd_25yd_Visita": 0, "Z4_25yd_ArcoVisita": 0}
}
registro_eventos = [] 

estado_posesion = "Indefinido" 
ultima_zona_valida = "Z2_25yd_50yd_Local" 

mostrar_anotaciones_ingeniero = True 
cambio_de_lado = False 

evidencia_posesion = 0.0      
UMBRAL_MOVIMIENTO = 0.05     
UMBRAL_EVIDENCIA_DINAMICO = 6 

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
    global estado_reproduccion, salto_solicitado, mostrar_anotaciones_ingeniero, cambio_de_lado, evidencia_posesion
    if event == cv2.EVENT_LBUTTONDOWN:
        f_actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if (VIDEO_H - 50) <= y <= VIDEO_H:
            if 10 <= x <= 70:    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f_actual - 5 * fps_video)); salto_solicitado = True
            elif 80 <= x <= 140: cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f_actual - 1 * fps_video)); salto_solicitado = True
            elif 150 <= x <= 230: estado_reproduccion = "PAUSE" if estado_reproduccion == "PLAY" else "PLAY"
            elif 240 <= x <= 300: cap.set(cv2.CAP_PROP_POS_FRAMES, f_actual + 1 * fps_video); salto_solicitado = True
            elif 310 <= x <= 370: cap.set(cv2.CAP_PROP_POS_FRAMES, f_actual + 5 * fps_video); salto_solicitado = True
            elif 380 <= x <= 560: mostrar_anotaciones_ingeniero = not mostrar_anotaciones_ingeniero
            elif 570 <= x <= 790: 
                cambio_de_lado = not cambio_de_lado
                evidencia_posesion = 0 

# ==========================================
# 4. LÓGICA SEMÁNTICA
# ==========================================
def obtener_metricas_caja(xyxy):
    x1, y1, x2, y2 = xyxy
    return (int((x1 + x2) / 2), int((y1 + y2) / 2), int(y2 - y1))

def inferir_zona_semantica(objetos_yolo, alto_pantalla, zona_previa, invertido):
    centro_accion = alto_pantalla / 2
    z_abajo = "Z4_25yd_ArcoVisita" if invertido else "Z1_ArcoLocal_25yd"
    z_arriba = "Z1_ArcoLocal_25yd" if invertido else "Z4_25yd_ArcoVisita"
    z_m_abajo = "Z3_50yd_25yd_Visita" if invertido else "Z2_25yd_50yd_Local"
    z_m_arriba = "Z2_25yd_50yd_Local" if invertido else "Z3_50yd_25yd_Visita"
    
    for obj in objetos_yolo:
        if obj["clase"] == "goal": return z_abajo if obj["cy"] > centro_accion else z_arriba
    for obj in objetos_yolo:
        if obj["clase"] == "50yd line": return z_m_arriba if obj["cy"] < centro_accion else z_m_abajo
    for obj in objetos_yolo:
        if obj["clase"] == "25yd line":
            if zona_previa in [z_abajo, z_m_abajo]: return z_abajo if obj["cy"] > centro_accion else z_m_abajo
            else: return z_m_arriba if obj["cy"] > centro_accion else z_arriba
                
    return "Zona_Transicion"

def inferir_zona_disputa(jugadores_xy, radio=120, min_jugadores=4):
    if len(jugadores_xy) < min_jugadores: return None, 0
    max_vecinos = 0; centro_disputa = None
    for (cx, cy, color) in jugadores_xy:
        vecinos = [(nx, ny) for (nx, ny, ncol) in jugadores_xy if np.sqrt((cx - nx)**2 + (cy - ny)**2) < radio]
        if len(vecinos) > max_vecinos:
            max_vecinos = len(vecinos)
            centro_disputa = (int(np.mean([v[0] for v in vecinos])), int(np.mean([v[1] for v in vecinos])))
    return (centro_disputa, max_vecinos) if max_vecinos >= min_jugadores else (None, 0)

# ==========================================
# 5. PIZARRA TÁCTICA 2D
# ==========================================
def dibujar_minimapa(lienzo, jugadores_info, centro_disputa, zona_actual, invertido):
    field_x1, field_y1 = VIDEO_W + 20, int(VIDEO_H * 0.05)
    field_w, field_h = PANEL_W - 40, int(VIDEO_H * 0.9)
    field_x2, field_y2 = field_x1 + field_w, field_y1 + field_h

    cv2.rectangle(lienzo, (VIDEO_W, 0), (TOTAL_W, VIDEO_H), (45, 115, 45), -1)
    
    field_border_color = (255, 255, 255)
    cv2.rectangle(lienzo, (field_x1, field_y1), (field_x2, field_y2), field_border_color, 2)
    
    cy_y = field_y1 + int(field_h * 0.5)
    cv2.line(lienzo, (field_x1, cy_y), (field_x2, cy_y), field_border_color, 2)
    cv2.circle(lienzo, (field_x1 + int(field_w*0.5), cy_y), int(field_w * 0.05), field_border_color, 1) 

    cv2.line(lienzo, (field_x1, field_y1 + int(field_h * 0.25)), (field_x2, field_y1 + int(field_h * 0.25)), (210, 210, 210), 1)
    cv2.line(lienzo, (field_x1, field_y1 + int(field_h * 0.75)), (field_x2, field_y1 + int(field_h * 0.75)), (210, 210, 210), 1)

    goal_circle_color = (250, 250, 250)
    goal_w = int(field_w * 0.6)
    goal_h = int(field_h * 0.1)
    
    gx1_vis = field_x1 + int(field_w*0.5 - goal_w*0.5)
    cv2.ellipse(lienzo, (field_x1 + int(field_w*0.5), field_y1), (goal_w // 2, goal_h), 0, 0, 180, goal_circle_color, 1) 
    cv2.rectangle(lienzo, (gx1_vis, field_y1), (gx1_vis + goal_w, field_y1 - 10), field_border_color, -1) 
    
    cv2.ellipse(lienzo, (field_x1 + int(field_w*0.5), field_y2), (goal_w // 2, int(goal_h*1.5)), 0, 180, 360, goal_circle_color, 1)
    cv2.rectangle(lienzo, (gx1_vis, field_y2), (gx1_vis + goal_w, field_y2 + 10), field_border_color, -1)

    cv2.putText(lienzo, "PIZARRA TACTICA 2D", (VIDEO_W + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cam_x, cam_y = field_x2 - 15, field_y1 + 15
    cv2.rectangle(lienzo, (cam_x - 10, cam_y - 5), (cam_x + 10, cam_y + 5), (200, 200, 200), 1) 
    cv2.rectangle(lienzo, (cam_x + 2, cam_y - 8), (cam_x + 8, cam_y + 8), (200, 200, 200), -1) 
    cv2.putText(lienzo, "HighBehind", (field_x2 - 80, field_y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    lbl_z4, lbl_z3 = ("Z1", "Z2") if invertido else ("Z4", "Z3")
    lbl_z2, lbl_z1 = ("Z3", "Z4") if invertido else ("Z2", "Z1")

    cv2.putText(lienzo, lbl_z4, (field_x1 - 25, field_y1 + int(field_h * 0.12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(lienzo, lbl_z3, (field_x1 - 25, field_y1 + int(field_h * 0.37)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(lienzo, lbl_z2, (field_x1 - 25, field_y1 + int(field_h * 0.62)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(lienzo, lbl_z1, (field_x1 - 25, field_y1 + int(field_h * 0.87)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.circle(lienzo, (field_x1, VIDEO_H - 15), 4, (255, 255, 255), -1) 
    cv2.putText(lienzo, "Claro", (field_x1 + 10, VIDEO_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.circle(lienzo, (field_x1 + 70, VIDEO_H - 15), 4, (0, 0, 0), -1) 
    cv2.putText(lienzo, "Oscuro", (field_x1 + 80, VIDEO_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    def proyectar_con_perspectiva(px, py):
        pct_x = px / VIDEO_W
        pct_y = py / VIDEO_H
        map_px = field_x1 + int(pct_x * field_w)
        map_py = field_y1 + int(pct_y * field_h)
        map_px = max(field_x1, min(field_x2, map_px))
        map_py = max(field_y1, min(field_y2, map_py))
        return map_px, map_py

    for (x, y, team_color) in jugadores_info:
        mx, my = proyectar_con_perspectiva(x, y)
        cv2.circle(lienzo, (mx, my), 4, team_color, -1) 
        
    if centro_disputa is not None:
        dx, dy = proyectar_con_perspectiva(centro_disputa[0], centro_disputa[1])
        cv2.circle(lienzo, (dx, dy), 12, (0, 165, 255), 1) 
        cv2.circle(lienzo, (dx, dy), 5, (0, 165, 255), -1) 

# ==========================================
# 6. INICIALIZACIÓN DE VENTANA
# ==========================================
cv2.namedWindow("Dashboard Analitico V8.1")
cv2.setMouseCallback("Dashboard Analitico V8.1", click_botones)
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
            evidencia_posesion = 0.0 
            salto_solicitado = False
        
        frame_resized = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        annotated_frame = frame_resized.copy()
        color_gris_actual = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        display_frame = np.zeros((VIDEO_H, TOTAL_W, 3), dtype=np.uint8)
        
        # --- A. CINEMÁTICA ---
        movimiento_y = 0.0
        if color_gris_previo is not None and puntos_previos is not None and len(puntos_previos) > 0:
            p_act, st, err = cv2.calcOpticalFlowPyrLK(color_gris_previo, color_gris_actual, puntos_previos, None, **lk_params)
            if p_act is not None:
                p_nuevos = p_act[st == 1]; p_viejos = puntos_previos[st == 1]
                if len(p_nuevos) > 0:
                    movimiento_y = np.mean(p_nuevos[:, 1] - p_viejos[:, 1])
                    if abs(movimiento_y) > UMBRAL_MOVIMIENTO: 
                        evidencia_posesion += (movimiento_y * 1.5) 
                    else:
                        evidencia_posesion *= 0.85 
                    evidencia_posesion = max(-25.0, min(25.0, evidencia_posesion))
        
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % 5 == 0 or puntos_previos is None or len(puntos_previos) < 10:
            puntos_previos = cv2.goodFeaturesToTrack(color_gris_actual, mask=None, **feature_params)
        color_gris_previo = color_gris_actual.copy()
        
        # --- B. IA Y EQUIPOS ---
        results = model.predict(frame_resized, conf=0.15, imgsz=800, agnostic_nms=True, verbose=False)
        objetos_detectados = []
        coords_jugadores_minimapa = [] 

        cajas_arcos_loop = [box.xyxy[0].cpu().numpy() for box in results[0].boxes if model.names[int(box.cls[0])].lower() == "goal"]

        for box in results[0].boxes:
            nombre_clase = model.names[int(box.cls[0])].lower()
            xyxy = box.xyxy[0].cpu().numpy()
            cx, cy, alto_box = obtener_metricas_caja(xyxy)
            
            if "line" in nombre_clase:
                if any((a[0]-15) < cx < (a[2]+15) and (a[1]-15) < cy < (a[3]+15) for a in cajas_arcos_loop): continue
            
            objetos_detectados.append({"clase": nombre_clase, "cy": cy, "alto_box": alto_box})
            
            if mostrar_anotaciones_ingeniero:
                if "goal" in nombre_clase: cv2.circle(annotated_frame, (cx, cy), 8, (255, 0, 0), -1) 
                elif "25yd line" in nombre_clase: cv2.circle(annotated_frame, (cx, cy), 8, (0, 255, 0), -1)

            if "player" in nombre_clase:
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(VIDEO_W, x2), min(VIDEO_H, y2)
                if x2 > x1 and y2 > y1:
                    player_hsv = cv2.cvtColor(frame_resized[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
                    team_color = (255, 255, 255) if np.mean(player_hsv[:, :, 2]) > 150 else (0, 0, 0)
                    coords_jugadores_minimapa.append((cx, cy, team_color))
                    if mostrar_anotaciones_ingeniero: cv2.circle(annotated_frame, (cx, cy), 3, (0, 255, 255), -1) 

        # --- C. DENSIDAD ---
        centro_disputa, cant_implicados = inferir_zona_disputa(coords_jugadores_minimapa, radio=120, min_jugadores=4)
        if mostrar_anotaciones_ingeniero and centro_disputa is not None:
            cv2.circle(annotated_frame, centro_disputa, 120, (0, 165, 255), 2)

        # --- D. MOTOR SEMÁNTICO Y CRONÓMETRO ---
        zona_actual_detectada = inferir_zona_semantica(objetos_detectados, VIDEO_H, ultima_zona_valida, cambio_de_lado)
        if zona_actual_detectada != "Zona_Transicion": ultima_zona_valida = zona_actual_detectada
            
        evento_trigger = None
        nuevo_estado = estado_posesion

        if evidencia_posesion >= UMBRAL_EVIDENCIA_DINAMICO:
            nuevo_estado = "Ataca_Arriba (Local)" if cambio_de_lado else "Ataca_Arriba (Visita)"
        elif evidencia_posesion <= -UMBRAL_EVIDENCIA_DINAMICO:
            nuevo_estado = "Ataca_Abajo (Visita)" if cambio_de_lado else "Ataca_Abajo (Local)"

        if nuevo_estado != estado_posesion:
            if estado_posesion != "Indefinido": 
                # Lógica Deductiva
                equipo_recuperador = "Local" if "Local" in nuevo_estado else "Visita"
                metricas_recuperacion[equipo_recuperador][ultima_zona_valida] += 1
                evento_trigger = f"RECUP. {equipo_recuperador.upper()} EN {ultima_zona_valida[:2]}"
                
                segundos_totales = int(cap.get(cv2.CAP_PROP_POS_FRAMES) / fps_video)
                minutos = segundos_totales // 60
                segundos = segundos_totales % 60
                tiempo_formateado = f"{minutos:02d}:{segundos:02d}"

                registro_eventos.append({
                    "Minuto_Video": tiempo_formateado,
                    "Equipo_Recuperador": equipo_recuperador,
                    "Zona_Recuperacion": ultima_zona_valida,
                    "Nuevo_Estado_Ataque": nuevo_estado,
                    "Cambio_Lado_Activo": cambio_de_lado,
                    "Frame_Exacto": int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                })
            estado_posesion = nuevo_estado
            evidencia_posesion = 0.0 

        # --- E. UI CON TABLERO AGRUPADO (DEFENSIVA / OFENSIVA) ---
        display_frame[0:VIDEO_H, 0:VIDEO_W] = annotated_frame if mostrar_anotaciones_ingeniero else frame_resized
        dibujar_minimapa(display_frame, coords_jugadores_minimapa, centro_disputa, ultima_zona_valida, cambio_de_lado)

        cv2.rectangle(display_frame, (0, 0), (VIDEO_W, 90), (15, 15, 15), -1)
        cv2.putText(display_frame, f"CINEMATICA: {estado_posesion} (Evid: {evidencia_posesion:.1f})", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        texto_zona = f"MEMORIA ZONA: {ultima_zona_valida}" if zona_actual_detectada == "Zona_Transicion" else f"VIENDO ZONA: {zona_actual_detectada}"
        color_zona = (0, 165, 255) if zona_actual_detectada == "Zona_Transicion" else (255, 255, 0)
        cv2.putText(display_frame, texto_zona, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_zona, 2)
        
        if 'evento_trigger' in locals() and evento_trigger:
            cv2.putText(display_frame, evento_trigger, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # TABLERO DE MÉTRICAS AGRUPADO (2 Líneas de alto en lugar de 4)
        cv2.putText(display_frame, "RECUPERACIONES ACUMULADAS", (VIDEO_W - 320, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        m_loc = metricas_recuperacion["Local"]
        m_vis = metricas_recuperacion["Visita"]

        # Columna Local (Izquierda)
        cv2.putText(display_frame, "LOCAL", (VIDEO_W - 320, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display_frame, f"Def: Z1:{m_loc['Z1_ArcoLocal_25yd']} Z2:{m_loc['Z2_25yd_50yd_Local']}", (VIDEO_W - 320, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(display_frame, f"Ata: Z3:{m_loc['Z3_50yd_25yd_Visita']} Z4:{m_loc['Z4_25yd_ArcoVisita']}", (VIDEO_W - 320, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Columna Visita (Derecha - Z4 y Z3 son su mitad defensiva)
        cv2.putText(display_frame, "VISITA", (VIDEO_W - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(display_frame, f"Def: Z4:{m_vis['Z4_25yd_ArcoVisita']} Z3:{m_vis['Z3_50yd_25yd_Visita']}", (VIDEO_W - 160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(display_frame, f"Ata: Z2:{m_vis['Z2_25yd_50yd_Local']} Z1:{m_vis['Z1_ArcoLocal_25yd']}", (VIDEO_W - 160, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.rectangle(display_frame, (0, VIDEO_H - 50), (VIDEO_W, VIDEO_H), (40, 40, 40), -1)
        
        botones = [("<< 5s", 10, 70), ("< 1s", 80, 140), (estado_reproduccion, 150, 230), ("> 1s", 240, 300), (">> 5s", 310, 370)]
        for texto, x1, x2 in botones:
            color = (0, 150, 0) if texto == "PLAY" else (0, 0, 150) if texto == "PAUSE" else (100, 100, 100)
            cv2.rectangle(display_frame, (x1, VIDEO_H - 40), (x2, VIDEO_H - 10), color, -1)
            cv2.putText(display_frame, texto, (x1 + 5, VIDEO_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        bx1, bx2 = 380, 560
        b_color = (130, 130, 30) if mostrar_anotaciones_ingeniero else (60, 60, 60)
        cv2.rectangle(display_frame, (bx1, VIDEO_H - 40), (bx2, VIDEO_H - 10), b_color, -1)
        cv2.putText(display_frame, "ANOTAC: " + ("ON" if mostrar_anotaciones_ingeniero else "OFF"), (bx1 + 5, VIDEO_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cx1, cx2 = 570, 790
        c_color = (130, 30, 130) if cambio_de_lado else (60, 60, 60)
        cv2.rectangle(display_frame, (cx1, VIDEO_H - 40), (cx2, VIDEO_H - 10), c_color, -1)
        cv2.putText(display_frame, "CAMBIO LADO: " + ("ON" if cambio_de_lado else "OFF"), (cx1 + 5, VIDEO_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Dashboard Analitico V8.1", display_frame)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    elif k == ord(' '): estado_reproduccion = "PAUSE" if estado_reproduccion == "PLAY" else "PLAY"

cap.release()
cv2.destroyAllWindows()

# ==========================================
# 8. EXPORTACIÓN DE RESULTADOS SEGURA
# ==========================================
import os

ruta_salida = r"C:\Users\ivang\Desktop\Tesis_Hockey\recuperaciones_hockey_final.csv"

df_resultados = pd.DataFrame(registro_eventos)

if not df_resultados.empty: 
    df_resultados = df_resultados[["Minuto_Video", "Equipo_Recuperador", "Zona_Recuperacion", "Nuevo_Estado_Ataque", "Cambio_Lado_Activo", "Frame_Exacto"]]
    df_resultados.to_csv(ruta_salida, index=False)

    print("\n" + "="*50)
    print("--- PROCESAMIENTO FINALIZADO CORRECTAMENTE ---")
    print("="*50)
    print(f"¡ÉXITO TÁCTICO! Se registraron {len(df_resultados)} recuperaciones.")
    print(f"EL ARCHIVO FUE CREADO EXACTAMENTE AQUÍ:\n>>> {ruta_salida} <<<")
    print("-" * 50)
    print(df_resultados.head(10))
else:
    print("ATENCIÓN: El video terminó o se presionó 'q', pero NO SE REGISTRARON CAMBIOS DE POSESIÓN.")