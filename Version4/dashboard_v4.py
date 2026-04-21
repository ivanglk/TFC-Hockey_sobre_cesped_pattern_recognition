import cv2
import numpy as np
from ultralytics import YOLO

print("--- INICIANDO PROTOTIPO: DASHBOARD TÁCTICO V4 (SEGUIMIENTO POR NODOS) ---")

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
# ⚠️ IMPORTANTE: Cambia esta ruta al nuevo best.pt de la V4
path_modelo = r"C:\Users\ivang\Desktop\Tesis_Hockey\models\best_v4.pt" 
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
# 3. FUNCIONES GEOMÉTRICAS V4 (Nodos)
# ==========================================
def obtener_centroide(box_xyxy):
    """Calcula el centro exacto (x, y) de una caja de YOLO."""
    x1, y1, x2, y2 = map(int, box_xyxy)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx, cy)

def ordenar_puntos(puntos):
    """Ordena 4 puntos para perspectiva de trapecio (Arriba->Abajo, Izq->Der)."""
    puntos = np.array(puntos)
    puntos_ordenados_y = puntos[np.argsort(puntos[:, 1])]
    
    arriba = puntos_ordenados_y[:2]
    abajo = puntos_ordenados_y[2:]
    
    tl = arriba[np.argsort(arriba[:, 0])][0] # Top-Left
    tr = arriba[np.argsort(arriba[:, 0])][1] # Top-Right
    bl = abajo[np.argsort(abajo[:, 0])][0]   # Bottom-Left
    br = abajo[np.argsort(abajo[:, 0])][1]   # Bottom-Right
    
    return np.float32([tl, tr, br, bl])

# ==========================================
# 4. CALIBRACIÓN INICIAL (Red de Seguridad)
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

print("Haz 4 CLICS demarcando la zona principal (ej. la mitad inferior de la cancha).")
cv2.imshow("Calibracion Inicial", frame_calibracion)
cv2.setMouseCallback("Calibracion Inicial", click_calibrador)
while len(puntos_video) < 4: cv2.waitKey(10)
cv2.destroyWindow("Calibracion Inicial")

src_base = ordenar_puntos(puntos_video)
dst_points = np.float32([
    [0, PIZARRA_H // 2],         # Top-Left 
    [PIZARRA_W, PIZARRA_H // 2], # Top-Right 
    [PIZARRA_W, PIZARRA_H],      # Bottom-Right 
    [0, PIZARRA_H]               # Bottom-Left 
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
    
    # Umbral de confianza ajustado para los nuevos nodos
    results = model.predict(frame_resized, conf=0.45, imgsz=800, verbose=False) ## Agregue imgz=800
    
    cajas_jugadores = []
    nodos_dinamicos = [] # Aquí guardaremos los centroides de arcos, T y L

    # A. EXTRACCIÓN SEMÁNTICA (El corazón de la V4)
    for box in results[0].boxes:
        clase = int(box.cls[0])
        nombre_clase = model.names[clase].lower()
        xyxy = box.xyxy[0]
        
        if "player" in nombre_clase:
            cajas_jugadores.append(box.xywh[0])
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        elif "goal" in nombre_clase:
            cx, cy = obtener_centroide(xyxy)
            nodos_dinamicos.append((cx, cy))
            cv2.circle(annotated_frame, (cx, cy), 8, (255, 0, 0), -1) # Arco = Azul
            cv2.putText(annotated_frame, "Ancla: Arco", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            
        elif "cruce_t" in nombre_clase:
            cx, cy = obtener_centroide(xyxy)
            nodos_dinamicos.append((cx, cy))
            cv2.circle(annotated_frame, (cx, cy), 8, (0, 0, 255), -1) # Cruce T = Rojo
            cv2.putText(annotated_frame, "Nodo: T", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            
        elif "cruce_l" in nombre_clase:
            cx, cy = obtener_centroide(xyxy)
            nodos_dinamicos.append((cx, cy))
            cv2.circle(annotated_frame, (cx, cy), 8, (0, 255, 255), -1) # Cruce L = Amarillo
            cv2.putText(annotated_frame, "Nodo: L", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

    # B. ACTUALIZACIÓN DE HOMOGRAFÍA
    if len(nodos_dinamicos) >= 4:
        puntos_extremos = ordenar_puntos(nodos_dinamicos[:4])
        try:
            matrix_H_activa = cv2.getPerspectiveTransform(puntos_extremos, dst_points)
            cv2.polylines(annotated_frame, [np.int32(puntos_extremos)], True, (255, 0, 0), 2)
            cv2.putText(annotated_frame, "MATRIZ: DINAMICA (Nodos V4)", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        except Exception:
            pass 
    else:
        cv2.polylines(annotated_frame, [np.int32(src_base)], True, (0, 255, 255), 2)
        cv2.putText(annotated_frame, "MATRIZ: ESTATICA (Memoria)", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # C. PROYECCIÓN DE JUGADORES
    for box_xywh in cajas_jugadores:
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