import cv2
import numpy as np
from ultralytics import YOLO
import os

print("--- INICIANDO PROTOTIPO: DASHBOARD TÁCTICO V3 (DEBUGGER DE VÉRTICES) ---")

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
path_modelo = r"C:\Users\ivang\Desktop\Tesis_Hockey\models\best_v3.pt"
path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\3raF-Inter_D_ 2024-VelezB_2-0SICb.mp4" 

VIDEO_W, VIDEO_H = 800, 600

# ==========================================
# 2. FUNCIONES MATEMÁTICAS
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
            w_caja = x2 - x1
            y_izq = int(m * (0 - x0) + y0)
            y_der = int(m * (w_caja - x0) + y0)
            return (x1, y1 + y_izq, x2, y1 + y_der), m
    return None, None

# ==========================================
# 3. MOTOR PRINCIPAL CON REPRODUCTOR
# ==========================================
model = YOLO(path_modelo)
cap = cv2.VideoCapture(path_video)

if not cap.isOpened():
    print("❌ ERROR: No se pudo abrir el video.")
    exit()

# Calcular cuántos frames son 1 segundo para los saltos
fps = cap.get(cv2.CAP_PROP_FPS)
frames_salto = int(fps) if fps > 0 else 30 

print("\nControles del reproductor en la ventana de video:")
print("[ESPACIO] o [P] -> Pausar / Reanudar")
print("[A] -> Rebobinar 1 segundo")
print("[D] -> Adelantar 1 segundo")
print("[Q] -> Salir\n")

is_paused = False
annotated_frame = None

while cap.isOpened():
    # Solo procesamos un nuevo frame si NO está pausado
    if not is_paused:
        success, frame = cap.read()
        if not success:
            print("Fin del video.")
            break

        frame_resized = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        annotated_frame = frame_resized.copy()
        
        results = model.predict(frame_resized, conf=0.3, verbose=False)
        
        lineas_transversales = [] 
        lineas_laterales = []     

        for box in results[0].boxes:
            clase = int(box.cls[0])
            nombre_clase = model.names[clase].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if nombre_clase in ["player", "goal"]:
                color = (0, 255, 0) if nombre_clase == "player" else (0, 165, 255)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
            elif "line" in nombre_clase:
                linea, pendiente = extraer_linea_matematica(frame_resized, x1, y1, x2, y2)
                
                if linea is not None and pendiente is not None:
                    lx1, ly1, lx2, ly2 = linea
                    
                    # ⚠️ HEURÍSTICA DE PENDIENTE (Posible culpable de los errores)
                    if abs(pendiente) < 0.5:
                        lineas_transversales.append(linea)
                        cv2.line(annotated_frame, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2) # Azul
                        cv2.putText(annotated_frame, f"m={pendiente:.2f}", (lx1, ly1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,0), 1)
                    else:
                        lineas_laterales.append(linea)
                        cv2.line(annotated_frame, (lx1, ly1), (lx2, ly2), (0, 0, 255), 2) # Roja
                        cv2.putText(annotated_frame, f"m={pendiente:.2f}", (lx1, ly1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

        # Buscar cruces y pintar vértices
        for l_trans in lineas_transversales:
            for l_lat in lineas_laterales:
                punto = calcular_interseccion(l_trans, l_lat)
                if punto:
                    px, py = punto
                    if -500 < px < VIDEO_W + 500 and -500 < py < VIDEO_H + 500:
                        cv2.circle(annotated_frame, (px, py), 6, (0, 255, 0), -1) 

    # Mostrar la imagen (se queda congelada si está pausado)
    cv2.imshow("Sistema Analitico - Vertices Automaticos", annotated_frame)

    # Lógica del teclado
    # Si está pausado, espera indefinidamente (0). Si reproduce, espera 1ms.
    tecla = cv2.waitKey(0 if is_paused else 1) & 0xFF

    if tecla == ord('q'):
        break
    elif tecla == ord('p') or tecla == 32: # 32 es la barra espaciadora
        is_paused = not is_paused
    elif tecla == ord('a'):
        # Rebobinar
        pos_actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos_actual - frames_salto))
        is_paused = False # Despausa para refrescar el frame
    elif tecla == ord('d'):
        # Adelantar
        pos_actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_actual + frames_salto)
        is_paused = False

cap.release()
cv2.destroyAllWindows()
print("--- FIN DE LA EJECUCIÓN ---")