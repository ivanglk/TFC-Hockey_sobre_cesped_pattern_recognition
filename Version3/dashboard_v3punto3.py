import cv2
import numpy as np
from ultralytics import YOLO
import os

print("--- INICIANDO PROTOTIPO: DASHBOARD TÁCTICO V3.1 (FILTROS TÁCTICOS ACTIVOS) ---")

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
# 3. MOTOR PRINCIPAL CON REPRODUCTOR Y FILTROS
# ==========================================
model = YOLO(path_modelo)
cap = cv2.VideoCapture(path_video)

if not cap.isOpened():
    print("❌ ERROR: No se pudo abrir el video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frames_salto = int(fps) if fps > 0 else 30 

print("\nControles del reproductor:")
print("[ESPACIO] o [P] -> Pausar / Reanudar")
print("[A] -> Rebobinar 1seg")
print("[D] -> Adelantar 1seg")
print("[Q] -> Salir\n")

is_paused = False
annotated_frame = None

while cap.isOpened():
    if not is_paused:
        success, frame = cap.read()
        if not success: break

        frame_resized = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        annotated_frame = frame_resized.copy()
        
        # ⚠️ SOLUCIÓN 1: Subimos la confianza de YOLO a 0.5 para filtrar ruido inicial
        results = model.predict(frame_resized, conf=0.5, verbose=False)
        
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
                    
                    # Heurística de pendiente
                    if abs(pendiente) < 0.5:
                        lineas_transversales.append(linea)
                        color_linea = (255, 0, 0) # Azul
                    else:
                        lineas_laterales.append(linea)
                        color_linea = (0, 0, 255) # Roja
                    
                    # Dibujar línea y pendiente
                    cv2.line(annotated_frame, (lx1, ly1), (lx2, ly2), color_linea, 2)
                    cv2.putText(annotated_frame, f"m={pendiente:.2f}", (lx1, ly1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_linea, 1)

        # ---------------------------------------------------------
        # ⚠️ SOLUCIÓN 2: FILTROS TÁCTICOS (ZONAS PROHIBIDAS)
        # ---------------------------------------------------------
        vertices_validos = []
        UMBRAL_LADO = 0.33 # Las laterales deben estar en el primer o último tercio
        
        for l_trans in lineas_transversales:
            for l_lat in lineas_laterales:
                
                # Calculamos el centro X de la línea lateral para saber si está a un costado
                lx1_l, ly1_l, lx2_l, ly2_l = l_lat
                centro_x_lateral = (lx1_l + lx2_l) // 2
                pos_relativa_x = centro_x_lateral / VIDEO_W
                
                punto = calcular_interseccion(l_trans, l_lat)
                
                if punto:
                    px, py = punto
                    
                    # REGLA TÁCTICA: Solo aceptamos intersecciones si la lateral está 
                    # a la IZQUIERDA (<33%) o a la DERECHA (>66%).
                    # Esto descarta la línea fantasma del medio.
                    if pos_relativa_x < UMBRAL_LADO or pos_relativa_x > (1 - UMBRAL_LADO):
                        # Además, debe estar dentro o cerca de la pantalla
                        if -VIDEO_W//2 < px < VIDEO_W * 1.5 and -VIDEO_H//2 < py < VIDEO_H * 1.5:
                            vertices_validos.append(punto)
                            # Pintamos en VERDE los vértices válidos
                            cv2.circle(annotated_frame, (px, py), 6, (0, 255, 0), -1) 
                    else:
                        # Opcional: Pintamos en AMARILLO los vértices descartados 
                        # por el filtro espacial para validar la tesis
                        cv2.circle(annotated_frame, (px, py), 4, (0, 255, 255), -1) 


    # Mostrar imagen
    cv2.imshow("Sistema Analitico - Debugger de Vertices v3.1", annotated_frame)

    # Lógica del teclado (igual que antes)
    tecla = cv2.waitKey(0 if is_paused else 1) & 0xFF
    if tecla == ord('q'):
        break
    elif tecla == ord('p') or tecla == 32:
        is_paused = not is_paused
    elif tecla == ord('a'):
        pos_actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, pos_actual - frames_salto))
        is_paused = False 
    elif tecla == ord('d'):
        pos_actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos_actual + frames_salto)
        is_paused = False

cap.release()
cv2.destroyAllWindows()
print("--- FIN DE LA EJECUCIÓN ---")