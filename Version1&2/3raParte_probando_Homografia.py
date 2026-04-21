import cv2
import numpy as np
from ultralytics import YOLO
import os

print("--- INICIANDO PROTOTIPO: DASHBOARD TÁCTICO V1 ---")

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS
# ==========================================
path_modelo = r"c:\Users\ivang\Desktop\Tesis_Hockey\models\best.pt"
path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\3raF-Inter_D_ 2024-VelezB_2-0SICb.mp4" 

# Resoluciones
VIDEO_W, VIDEO_H = 800, 600
PIZARRA_W, PIZARRA_H = 350, 600

# ==========================================
# 2. FUNCIONES DE DIBUJO Y LÓGICA
# ==========================================
def crear_pizarra_hockey(alto=600, ancho=350):
    """Dibuja una cancha de hockey vertical estilizada."""
    pizarra = np.zeros((alto, ancho, 3), dtype=np.uint8)
    pizarra[:] = (40, 110, 40) # Verde táctico
    
    color_linea = (255, 255, 255)
    grosor = 2
    medio_y = alto // 2
    linea_23_sup = int(alto * 0.25)
    linea_23_inf = int(alto * 0.75)
    radio_area = int(ancho * 0.3)
    
    cv2.rectangle(pizarra, (0, 0), (ancho-1, alto-1), color_linea, grosor)
    cv2.line(pizarra, (0, medio_y), (ancho, medio_y), color_linea, grosor)
    cv2.line(pizarra, (0, linea_23_sup), (ancho, linea_23_sup), color_linea, grosor)
    cv2.line(pizarra, (0, linea_23_inf), (ancho, linea_23_inf), color_linea, grosor)
    
    centro_arco_sup = (ancho // 2, 0)
    centro_arco_inf = (ancho // 2, alto)
    cv2.ellipse(pizarra, centro_arco_sup, (radio_area, radio_area), 0, 0, 180, color_linea, grosor)
    cv2.ellipse(pizarra, centro_arco_inf, (radio_area, radio_area), 0, 180, 360, color_linea, grosor)
    
    cv2.circle(pizarra, (ancho // 2, int(alto * 0.12)), 3, color_linea, -1)
    cv2.circle(pizarra, (ancho // 2, int(alto * 0.88)), 3, color_linea, -1)
    return pizarra

def clasificar_equipo(frame, box):
    """Filtra el pasto y clasifica al jugador por la luminosidad de su camiseta."""
    x_centro, y_centro, w, h = box
    
    x1, x2 = int(max(0, x_centro - w/4)), int(min(frame.shape[1], x_centro + w/4))
    y1, y2 = int(max(0, y_centro - h/2)), int(min(frame.shape[0], y_centro))
    
    torso = frame[y1:y2, x1:x2]
    if torso.size == 0:
        return (128, 128, 128), "Desconocido"
        
    hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    mask_no_pasto = cv2.inRange(hsv_torso, (0, 0, 0), (34, 255, 255)) | \
                    cv2.inRange(hsv_torso, (86, 0, 0), (179, 255, 255))
    
    color_bgr = cv2.mean(torso, mask=mask_no_pasto)[:3]
    luminosidad = color_bgr[0]*0.114 + color_bgr[1]*0.587 + color_bgr[2]*0.299
    
    # Ajusta este valor si los equipos se confunden (100 suele ir bien para Blanco vs Oscuro)
    UMBRAL_LUMINOSIDAD = 100 
    
    if luminosidad > UMBRAL_LUMINOSIDAD:
        return (255, 255, 255), "Equipo Claro" 
    else:
        return (255, 0, 0), "Equipo Oscuro" 

# ==========================================
# 3. MÓDULO DE CALIBRACIÓN MANUAL (HOMOGRAFÍA ESTÁTICA)
# ==========================================
model = YOLO(path_modelo)
cap = cv2.VideoCapture(path_video)

if not cap.isOpened():
    print("❌ ERROR: No se pudo abrir el video.")
    exit()

success, frame_calibracion = cap.read()
frame_calibracion = cv2.resize(frame_calibracion, (VIDEO_W, VIDEO_H))

puntos_video = []
def click_calibrador(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(puntos_video) < 4:
        puntos_video.append([x, y])
        cv2.circle(frame_calibracion, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(frame_calibracion, str(len(puntos_video)), (x+10, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
        cv2.imshow("Calibracion Inicial", frame_calibracion)

print("\n--- PASO 1: CALIBRACIÓN DE CÁMARA ---")
print("Haz 4 CLICS en la ventana para demarcar la mitad inferior de la cancha.")
print("Orden estricto:")
print("1. Línea Central - Borde Izquierdo")
print("2. Línea Central - Borde Derecho")
print("3. Línea de Fondo (Arco abajo) - Borde Derecho")
print("4. Línea de Fondo (Arco abajo) - Borde Izquierdo\n")

cv2.imshow("Calibracion Inicial", frame_calibracion)
cv2.setMouseCallback("Calibracion Inicial", click_calibrador)

# Esperar hasta tener los 4 puntos
while len(puntos_video) < 4:
    cv2.waitKey(10)

cv2.destroyWindow("Calibracion Inicial")

# Calcular la Matriz H
src_points = np.float32(puntos_video)
dst_points = np.float32([
    [0, PIZARRA_H // 2],           # 1. Medio Izquierda
    [PIZARRA_W, PIZARRA_H // 2],   # 2. Medio Derecha
    [PIZARRA_W, PIZARRA_H],        # 3. Fondo Derecha
    [0, PIZARRA_H]                 # 4. Fondo Izquierda
])
matrix_H = cv2.getPerspectiveTransform(src_points, dst_points)
print("✅ Matriz de Homografía H calculada. Iniciando análisis...\n")

# ==========================================
# 4. BUCLE PRINCIPAL DE PROCESAMIENTO
# ==========================================
print("🎥 Dashboard en ejecución... (Presiona 'q' para salir)")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_resized = cv2.resize(frame, (VIDEO_W, VIDEO_H))
    results = model.predict(frame_resized, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()
    
    mapa_2d = crear_pizarra_hockey(alto=PIZARRA_H, ancho=PIZARRA_W)

    # Procesar detecciones
    for box in results[0].boxes:
        clase = int(box.cls[0])
        nombre_clase = model.names[clase]
        
        if nombre_clase.lower() == "player":
            x_centro, y_centro, w, h = box.xywh[0]
            x_suelo = float(x_centro)
            y_suelo = float(y_centro + (h / 2)) # Base de los pies
            
            # Aplicar Homografía Real
            punto_base = np.array([[[x_suelo, y_suelo]]], dtype=np.float32)
            punto_transformado = cv2.perspectiveTransform(punto_base, matrix_H)
            
            map_x = int(punto_transformado[0][0][0])
            map_y = int(punto_transformado[0][0][1])
            
            # Dibujar solo si el jugador cae dentro del área de la pizarra
            if 0 <= map_x <= PIZARRA_W and 0 <= map_y <= PIZARRA_H:
                color_equipo, _ = clasificar_equipo(frame_resized, box.xywh[0])
                cv2.circle(mapa_2d, (map_x, map_y), 7, color_equipo, -1)
                cv2.circle(mapa_2d, (map_x, map_y), 7, (0, 0, 0), 1)

    # Construir Dashboard
    dashboard = np.hstack((annotated_frame, mapa_2d))
    cv2.rectangle(dashboard, (0, 0), (dashboard.shape[1], 40), (25, 25, 25), -1)
    cv2.putText(dashboard, "TESIS HOCKEY - Dashboard Tactico Espacial", (20, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Sistema Analitico", dashboard)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("--- FIN DE LA EJECUCIÓN ---")