import cv2
import numpy as np
from ultralytics import YOLO

print("--- INICIANDO PROTOTIPO: DASHBOARD TÁCTICO V4.1 (FIX HOMOGRAFÍA) ---")

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
path_modelo = r"C:\Users\ivang\Desktop\Tesis_Hockey\models\best_v3.pt"
path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\3raF-Inter_D_ 2024-VelezB_2-0SICb.mp4" 

VIDEO_W, VIDEO_H = 800, 600
PIZARRA_W, PIZARRA_H = 350, 600

# ==========================================
# 2. LÓGICA 2D Y CLASIFICACIÓN
# ==========================================
def crear_pizarra_hockey(alto=600, ancho=350):
    pizarra = np.zeros((alto, ancho, 3), dtype=np.uint8)
    pizarra[:] = (40, 110, 40) 
    color_linea = (255, 255, 255)
    grosor = 2
    medio_y = alto // 2
    linea_23_sup, linea_23_inf = int(alto * 0.25), int(alto * 0.75)
    radio_area = int(ancho * 0.3)
    
    cv2.rectangle(pizarra, (0, 0), (ancho-1, alto-1), color_linea, grosor)
    cv2.line(pizarra, (0, medio_y), (ancho, medio_y), color_linea, grosor)
    cv2.line(pizarra, (0, linea_23_sup), (ancho, linea_23_sup), color_linea, grosor)
    cv2.line(pizarra, (0, linea_23_inf), (ancho, linea_23_inf), color_linea, grosor)
    
    cv2.ellipse(pizarra, (ancho // 2, 0), (radio_area, radio_area), 0, 0, 180, color_linea, grosor)
    cv2.ellipse(pizarra, (ancho // 2, alto), (radio_area, radio_area), 0, 180, 360, color_linea, grosor)
    return pizarra

def clasificar_equipo(frame, box_xywh):
    # Aseguramos que los valores sean floats y no tensores de IA
    x_c, y_c, w, h = [float(v) for v in box_xywh]
    
    x1, x2 = int(max(0, x_c - w/4)), int(min(frame.shape[1], x_c + w/4))
    y1, y2 = int(max(0, y_c - h/2)), int(min(frame.shape[0], y_c))
    torso = frame[y1:y2, x1:x2]
    
    if torso.size == 0: return (128, 128, 128)
        
    hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    mask_no_pasto = cv2.inRange(hsv_torso, (0,0,0), (34,255,255)) | cv2.inRange(hsv_torso, (86,0,0), (179,255,255))
    color_bgr = cv2.mean(torso, mask=mask_no_pasto)[:3]
    luminosidad = color_bgr[0]*0.114 + color_bgr[1]*0.587 + color_bgr[2]*0.299
    
    return (255, 255, 255) if luminosidad > 100 else (255, 0, 0)

# ==========================================
# 3. FUNCIONES GEOMÉTRICAS 
# ==========================================
def calcular_interseccion(linea1, linea2):
    x1, y1, x2, y2 = linea1
    x3, y3, x4, y4 = linea2
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if D == 0: return None
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / D
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / D
    return (int(px), int(py))

def extraer_linea_matematica(frame, x1, y1, x2, y2):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return None, None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gray, 50, 150)
    puntos_blancos = cv2.findNonZero(bordes)
    if puntos_blancos is not None and len(puntos_blancos) > 30:
        [vx, vy, x0, y0] = cv2.fitLine(puntos_blancos, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]
        if abs(vx) > 0.001:
            m = vy / vx 
            y_izq = int(m * (0 - x0) + y0)
            y_der = int(m * ((x2-x1) - x0) + y0)
            return (x1, y1 + y_izq, x2, y1 + y_der), m
    return None, None

def ordenar_puntos(puntos):
    """Ordena 4 puntos para perspectiva deportiva (Trapecio)."""
    puntos = np.array(puntos)
    
    # 1. Ordenar los 4 puntos por su eje Y (de arriba hacia abajo en la pantalla)
    puntos_ordenados_y = puntos[np.argsort(puntos[:, 1])]
    
    # 2. Los 2 primeros son los de ARRIBA, los 2 últimos son los de ABAJO
    arriba = puntos_ordenados_y[:2]
    abajo = puntos_ordenados_y[2:]
    
    # 3. Para los de arriba, ordenamos por X (Izquierda a Derecha)
    tl = arriba[np.argsort(arriba[:, 0])][0] # Top-Left
    tr = arriba[np.argsort(arriba[:, 0])][1] # Top-Right
    
    # 4. Para los de abajo, ordenamos por X (Izquierda a Derecha)
    bl = abajo[np.argsort(abajo[:, 0])][0]   # Bottom-Left
    br = abajo[np.argsort(abajo[:, 0])][1]   # Bottom-Right
    
    # Devolvemos en el orden estricto que exige OpenCV: Arriba-Izq, Arriba-Der, Abajo-Der, Abajo-Izq
    return np.float32([tl, tr, br, bl])

# ==========================================
# 4. CALIBRACIÓN INICIAL
# ==========================================
model = YOLO(path_modelo)
cap = cv2.VideoCapture(path_video)

success, frame_calibracion = cap.read()
frame_calibracion = cv2.resize(frame_calibracion, (VIDEO_W, VIDEO_H))

puntos_video = []
def click_calibrador(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(puntos_video) < 4:
        puntos_video.append([x, y])
        cv2.circle(frame_calibracion, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Calibracion Inicial", frame_calibracion)

print("Haz 4 CLICS demarcando la mitad inferior de la cancha (formando un rectángulo/trapecio).")
cv2.imshow("Calibracion Inicial", frame_calibracion)
cv2.setMouseCallback("Calibracion Inicial", click_calibrador)
while len(puntos_video) < 4: cv2.waitKey(10)
cv2.destroyWindow("Calibracion Inicial")

# EL FIX MÁGICO 1: Ordenamos tus clics matemáticamente
src_base = ordenar_puntos(puntos_video)

# Mapeamos a la mitad inferior de la pizarra 2D
dst_points = np.float32([
    [0, PIZARRA_H // 2],         # Top-Left (Medio campo)
    [PIZARRA_W, PIZARRA_H // 2], # Top-Right (Medio campo)
    [PIZARRA_W, PIZARRA_H],      # Bottom-Right (Fondo)
    [0, PIZARRA_H]               # Bottom-Left (Fondo)
])
matrix_H_activa = cv2.getPerspectiveTransform(src_base, dst_points)

# ==========================================
# 5. BUCLE PRINCIPAL DINÁMICO
# ==========================================
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame_resized = cv2.resize(frame, (VIDEO_W, VIDEO_H))
    annotated_frame = frame_resized.copy()
    mapa_2d = crear_pizarra_hockey(PIZARRA_H, PIZARRA_W)
    
    results = model.predict(frame_resized, conf=0.5, verbose=False)
    
    lineas_trans, lineas_lat, cajas_jugadores = [], [], []

    # Recolección
    for box in results[0].boxes:
        clase = int(box.cls[0])
        nombre_clase = model.names[clase].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        if nombre_clase in ["player"]:
            cajas_jugadores.append(box.xywh[0])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        elif "line" in nombre_clase:
            linea, pendiente = extraer_linea_matematica(frame_resized, x1, y1, x2, y2)
            if linea and pendiente is not None:
                if abs(pendiente) < 0.5: lineas_trans.append(linea)
                else: lineas_lat.append(linea)

    # Buscamos vértices válidos
    vertices = []
    for l_trans in lineas_trans:
        for l_lat in lineas_lat:
            cx_lat = (l_lat[0] + l_lat[2]) // 2
            if (cx_lat / VIDEO_W) < 0.33 or (cx_lat / VIDEO_W) > 0.66:
                pt = calcular_interseccion(l_trans, l_lat)
                if pt and -200 < pt[0] < VIDEO_W+200 and -200 < pt[1] < VIDEO_H+200:
                    vertices.append(pt)

    # Dinámica vs Estática
    if len(vertices) >= 4:
        puntos_extremos = ordenar_puntos(vertices[:4])
        try:
            matrix_H_activa = cv2.getPerspectiveTransform(puntos_extremos, dst_points)
            # Dibujamos en azul el área mapeada dinámica
            cv2.polylines(annotated_frame, [np.int32(puntos_extremos)], True, (255, 0, 0), 2)
            cv2.putText(annotated_frame, "MATRIZ: DINAMICA", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception:
            pass 
    else:
        # Dibujamos en amarillo el área mapeada estática (la de tus clics)
        cv2.polylines(annotated_frame, [np.int32(src_base)], True, (0, 255, 255), 2)
        cv2.putText(annotated_frame, "MATRIZ: ESTATICA (Memoria)", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # D. PROYECCIÓN DE JUGADORES AL MAPA 2D
    for box_xywh in cajas_jugadores:
        # EL FIX MÁGICO 2: Limpieza del tensor de IA a Python puro
        x_c, y_c, w, h = [float(v) for v in box_xywh]
        y_suelo = y_c + (h / 2)
        
        punto_base = np.array([[[x_c, y_suelo]]], dtype=np.float32)
        punto_2d = cv2.perspectiveTransform(punto_base, matrix_H_activa)
        
        map_x, map_y = int(punto_2d[0][0][0]), int(punto_2d[0][0][1])
        
        if 0 <= map_x <= PIZARRA_W and 0 <= map_y <= PIZARRA_H:
            color_equipo = clasificar_equipo(frame_resized, box_xywh)
            cv2.circle(mapa_2d, (map_x, map_y), 7, color_equipo, -1)
            cv2.circle(mapa_2d, (map_x, map_y), 7, (0,0,0), 1)

    dashboard = np.hstack((annotated_frame, mapa_2d))
    cv2.imshow("Sistema Analitico", dashboard)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()