import cv2
import numpy as np
from ultralytics import YOLO

print("--- INICIANDO SISTEMA HÍBRIDO V4: IA + INFERENCIA MATEMÁTICA ---")

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
# ⚠️ Actualiza esta ruta mañana cuando descargues tu nuevo modelo
path_modelo = r"C:\Users\ivang\Desktop\Tesis_Hockey\models\best_v5.pt" ##el hibrido
path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\6taFecha-2024-Vélez _B_ 2-0DAOM.mp4" 

VIDEO_W, VIDEO_H = 800, 600

# ==========================================
# 2. MOTOR MATEMÁTICO (El corazón híbrido)
# ==========================================
def obtener_centroide(xyxy):
    """Calcula el centro de una caja detectada por YOLO."""
    return (int((xyxy[0] + xyxy[2]) / 2), int((xyxy[1] + xyxy[3]) / 2))

def extraer_lateral_por_color(frame, box_xyxy):
    """
    Encuentra la línea lateral buscando el contraste entre el pasto verde
    y el color del exterior de la cancha usando máscaras HSV.
    """
    x1, y1, x2, y2 = map(int, box_xyxy)
    
    # 1. Recortar la caja donde YOLO nos dijo que está la "Lateral Line"
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return None
    
    # 2. Convertir a espacio de color HSV (inmune a sombras)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 3. Definir el rango del "Verde Cancha" (Ajustable según la iluminación de tu video)
    # H: Matiz (35 a 85 abarca amarillentos a verdes oscuros)
    # S: Saturación (40 a 255 ignora grises/blancos)
    # V: Brillo (40 a 255 ignora sombras negras profundas)
    rango_bajo_verde = np.array([35, 40, 40])
    rango_alto_verde = np.array([85, 255, 255])
    
    # Crear una máscara: Blanco donde hay pasto, Negro donde está el exterior
    mascara_cancha = cv2.inRange(hsv, rango_bajo_verde, rango_alto_verde)
    
    # Opcional: Limpiar el ruido de los jugadores pisando la línea
    kernel = np.ones((5,5), np.uint8)
    mascara_limpia = cv2.morphologyEx(mascara_cancha, cv2.MORPH_CLOSE, kernel)
    
    # 4. El borde exacto de esta máscara ES nuestra línea lateral
    edges = cv2.Canny(mascara_limpia, 50, 150)
    pts = cv2.findNonZero(edges)
    
    # 5. Si encontramos un borde largo, tiramos la regresión lineal
    if pts is not None and len(pts) > 20:
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        return (float(vx[0]), float(vy[0]), float(x0[0] + x1), float(y0[0] + y1))
    
    return None

def extraer_linea_hough(frame, box_xyxy):
    """Extrae la línea lateral usando Transformada de Hough (Inmune a la basura exterior)."""
    try:
        alto_frame, ancho_frame = frame.shape[:2]
        
        # 1. CLAMPING
        x1 = max(0, int(box_xyxy[0]))
        y1 = max(0, int(box_xyxy[1]))
        x2 = min(ancho_frame, int(box_xyxy[2]))
        y2 = min(alto_frame, int(box_xyxy[3]))
        
        if x2 <= x1 or y2 <= y1: return None
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return None
            
        # 2. FILTRO DE CAL (Máscara HSV Blanca)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        blanco_bajo = np.array([0, 0, 160]) 
        blanco_alto = np.array([179, 50, 255])
        mascara_blanca = cv2.inRange(hsv, blanco_bajo, blanco_alto)
        
        edges = cv2.Canny(mascara_blanca, 30, 100) 
        
        # 3. TRANSFORMADA DE HOUGH (El cambio clave)
        # Busca segmentos rectos de al menos 40 píxeles de largo
        lineas = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=40, maxLineGap=20)
        
        if lineas is not None and len(lineas) > 0:
            # Buscar la línea estructurada más larga, ignorando el ruido
            linea_mas_larga = None
            max_longitud = 0
            
            for linea in lineas:
                xl1, yl1, xl2, yl2 = linea[0]
                longitud = np.sqrt((xl2 - xl1)**2 + (yl2 - yl1)**2)
                if longitud > max_longitud:
                    max_longitud = longitud
                    linea_mas_larga = linea[0]
            
            if linea_mas_larga is not None:
                xl1, yl1, xl2, yl2 = linea_mas_larga
                # Convertir los 2 puntos de Hough al formato (vector_x, vector_y, punto_x, punto_y)
                vx = xl2 - xl1
                vy = yl2 - yl1
                norma = np.sqrt(vx**2 + vy**2)
                if norma == 0: return None
                
                # Devolvemos la línea re-mapeada a la pantalla completa
                return (float(vx/norma), float(vy/norma), float(xl1 + x1), float(yl1 + y1))
                
        return None
        
    except Exception:
        return None

def extraer_recta_matematica(frame, box_xyxy):
    """Extrae la línea aplicando primero un filtro estricto para el color BLANCO."""
    try:
        alto_frame, ancho_frame = frame.shape[:2]
        
        # 1. CLAMPING (Seguridad)
        x1 = max(0, int(box_xyxy[0]))
        y1 = max(0, int(box_xyxy[1]))
        x2 = min(ancho_frame, int(box_xyxy[2]))
        y2 = min(alto_frame, int(box_xyxy[3]))
        
        if x2 <= x1 or y2 <= y1: return None
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return None
            
        # 2. MÁSCARA DE COLOR (El Filtro de Cal)
        # Convertimos a HSV porque es mucho más resistente a los cambios de luz
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Rango para el color BLANCO en HSV
        # Baja saturación (0-40) y Alto brillo (180-255)
        blanco_bajo = np.array([0, 0, 160]) 
        blanco_alto = np.array([179, 50, 255])
        
        # Creamos una imagen donde SOLO la pintura blanca es visible (lo demás es negro)
        mascara_blanca = cv2.inRange(hsv, blanco_bajo, blanco_alto)
        
        # 3. MATEMÁTICA SOBRE LA MÁSCARA
        # Bajamos los umbrales de Canny para que no parpadee tanto con el blur
        edges = cv2.Canny(mascara_blanca, 30, 100) 
        pts = cv2.findNonZero(edges)
        
        # 4. EXTRACCIÓN
        if pts is not None and len(pts) > 15:
            linea = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = linea[0], linea[1], linea[2], linea[3]
            return (float(vx[0]), float(vy[0]), float(x0[0] + x1), float(y0[0] + y1))
            
        return None
        
    except Exception:
        return None

def calcular_interseccion(recta1, recta2):
    """Calcula la colisión (X,Y) entre dos rectas matemáticas usando determinantes."""
    if recta1 is None or recta2 is None: return None
    
    vx1, vy1, x1, y1 = recta1
    vx2, vy2, x2, y2 = recta2
    
    # Evitar líneas paralelas (determinante cero)
    det = (vx1 * vy2) - (vy1 * vx2)
    if abs(det) < 1e-6: return None
    
    # Álgebra de intersección de líneas 2D
    t1 = ((x2 - x1) * vy2 - (y2 - y1) * vx2) / det
    pto_colision_x = int(x1 + t1 * vx1)
    pto_colision_y = int(y1 + t1 * vy1)
    
    return (pto_colision_x, pto_colision_y)

# ==========================================
# 3. BUCLE PRINCIPAL
# ==========================================
model = YOLO(path_modelo)
cap = cv2.VideoCapture(path_video)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame_resized = cv2.resize(frame, (VIDEO_W, VIDEO_H))
    annotated_frame = frame_resized.copy()
    
    # ⚠️ Umbral bajo (0.20) para que no se le escapen las líneas lejanas
    # results = model.predict(frame_resized, conf=0.20, imgsz=800, verbose=False)
    # Bajamos el umbral a 0.15 y forzamos resolución de 800 para ver detalles en el zoom
    results = model.predict(frame_resized, conf=0.15, imgsz=800, agnostic_nms=True, verbose=False)
    
    nodos_dinamicos = []
    cajas_lineas = {"25yd": None, "lateral": None}

    # FASE A: Lectura directa de la IA
    for box in results[0].boxes:
        clase = int(box.cls[0])
        nombre_clase = model.names[clase].lower()
        xyxy = box.xyxy[0].cpu().numpy()
        
        if "goal" in nombre_clase:
            cx, cy = obtener_centroide(xyxy)
            nodos_dinamicos.append((cx, cy))
            cv2.circle(annotated_frame, (cx, cy), 8, (255, 0, 0), -1) # Azul = Arco
            
        elif "cruce_t" in nombre_clase:
            cx, cy = obtener_centroide(xyxy)
            nodos_dinamicos.append((cx, cy))
            cv2.circle(annotated_frame, (cx, cy), 8, (0, 0, 255), -1) # Rojo = T directa
            
        # Guardamos las cajas grandes para la matemática
        elif "25yd line" in nombre_clase:
            cajas_lineas["25yd"] = xyxy
        elif "lateral line" in nombre_clase:
            cajas_lineas["lateral"] = xyxy


   # FASE B: Inferencia Matemática (Modo DEBUG VISUAL)
    # FASE B: Inferencia Matemática (Modo DEBUG VISUAL)
    if cajas_lineas["25yd"] is not None and cajas_lineas["lateral"] is not None:
        
        # ¡Ambas líneas ahora usan el poder de Hough!
        recta_25 = extraer_linea_hough(frame_resized, cajas_lineas["25yd"])
        recta_lat = extraer_linea_hough(frame_resized, cajas_lineas["lateral"])
        
        # --- INICIO DE DEBUG VISUAL ---
        if recta_25 is not None:
            vx, vy, x0, y0 = recta_25
            pt1 = (int(x0 - 2000*vx), int(y0 - 2000*vy))
            pt2 = (int(x0 + 2000*vx), int(y0 + 2000*vy))
            cv2.line(annotated_frame, pt1, pt2, (255, 255, 0), 2) # Celeste
            
        if recta_lat is not None:
            vx, vy, x0, y0 = recta_lat
            pt1 = (int(x0 - 2000*vx), int(y0 - 2000*vy))
            pt2 = (int(x0 + 2000*vx), int(y0 + 2000*vy))
            cv2.line(annotated_frame, pt1, pt2, (255, 0, 255), 2) # Violeta
        # --- FIN DE DEBUG VISUAL ---

        nodo_virtual = calcular_interseccion(recta_25, recta_lat)
        
        if nodo_virtual is not None:
            if 0 <= nodo_virtual[0] <= VIDEO_W and 0 <= nodo_virtual[1] <= VIDEO_H:
                nodos_dinamicos.append(nodo_virtual)
                cv2.circle(annotated_frame, nodo_virtual, 10, (0, 0, 255), 2)
                cv2.putText(annotated_frame, "T VIRTUAL", (nodo_virtual[0]+12, nodo_virtual[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        


    cv2.imshow("Dashboard Hibrido V4", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()