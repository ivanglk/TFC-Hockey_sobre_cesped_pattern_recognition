import cv2
import numpy as np
from ultralytics import YOLO
import os

print("--- INICIANDO PROTOTIPO: DASHBOARD TÁCTICO V3 (BUSCADOR DE VÉRTICES) ---")

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
path_modelo = r"C:\Users\ivang\Desktop\Tesis_Hockey\models\best_v3.pt"
path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\3raF-Inter_D_ 2024-VelezB_2-0SICb.mp4" 

VIDEO_W, VIDEO_H = 800, 600

# ==========================================
# 2. FUNCIONES MATEMÁTICAS (El Laboratorio Integrado)
# ==========================================
def calcular_interseccion(linea1, linea2):
    """Calcula el punto de intersección (x, y) de dos rectas."""
    x1, y1, x2, y2 = linea1
    x3, y3, x4, y4 = linea2
    
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if D == 0:
        return None
    
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / D
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / D
    return (int(px), int(py))

def extraer_linea_matematica(frame, x1, y1, x2, y2):
    """Aplica Canny y Regresión Lineal dentro de la caja de YOLO."""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return None, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(gray, 50, 150)
    puntos_blancos = cv2.findNonZero(bordes)

    if puntos_blancos is not None and len(puntos_blancos) > 30:
        [vx, vy, x0, y0] = cv2.fitLine(puntos_blancos, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

        if abs(vx) > 0.001:
            m = vy / vx # Pendiente
            w_caja = x2 - x1
            y_izq = int(m * (0 - x0) + y0)
            y_der = int(m * (w_caja - x0) + y0)
            
            # Devolvemos las coordenadas absolutas de la línea y su pendiente
            return (x1, y1 + y_izq, x2, y1 + y_der), m
    return None, None

# ==========================================
# 3. MOTOR PRINCIPAL
# ==========================================
model = YOLO(path_modelo)
cap = cv2.VideoCapture(path_video)

if not cap.isOpened():
    print("❌ ERROR: No se pudo abrir el video.")
    exit()

print("🎥 Buscando vértices en tiempo real... (Presiona 'q' para salir)")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_resized = cv2.resize(frame, (VIDEO_W, VIDEO_H))
    annotated_frame = frame_resized.copy()
    
    results = model.predict(frame_resized, conf=0.3, verbose=False)
    
    # Listas para guardar las líneas clasificadas matemáticamente en este fotograma
    lineas_transversales = [] # Fondo, 25, 50
    lineas_laterales = []     # Laterales

    for box in results[0].boxes:
        clase = int(box.cls[0])
        nombre_clase = model.names[clase].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 1. Dibujar Jugadores/Arcos (Para el DT)
        if nombre_clase in ["player", "goal"]:
            color = (0, 255, 0) if nombre_clase == "player" else (0, 165, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
        # 2. Procesar Líneas Geométricas (En segundo plano)
        elif "line" in nombre_clase:
            linea, pendiente = extraer_linea_matematica(frame_resized, x1, y1, x2, y2)
            
            if linea is not None and pendiente is not None:
                lx1, ly1, lx2, ly2 = linea
                
                # CLASIFICACIÓN HEURÍSTICA POR PENDIENTE
                # Si la pendiente es menor a 0.5 (ángulo suave), es transversal (fondo, 25, 50)
                if abs(pendiente) < 0.5:
                    lineas_transversales.append(linea)
                    cv2.line(annotated_frame, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2) # Azul
                # Si es muy empinada, es lateral
                else:
                    lineas_laterales.append(linea)
                    cv2.line(annotated_frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2) # Roja

    # 3. CALCULAR COLISIONES (Los 10 vértices teóricos)
    for l_trans in lineas_transversales:
        for l_lat in lineas_laterales:
            punto = calcular_interseccion(l_trans, l_lat)
            
            if punto:
                px, py = punto
                # Dibujamos el vértice solo si cae dentro o un poco afuera de la pantalla
                if -200 < px < VIDEO_W + 200 and -200 < py < VIDEO_H + 200:
                    cv2.circle(annotated_frame, (px, py), 6, (0, 255, 0), -1) # Punto Verde

    cv2.imshow("Sistema Analitico - Vertices Automaticos", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("--- FIN DE LA EJECUCIÓN ---")