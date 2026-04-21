import cv2
import numpy as np
from ultralytics import YOLO

def calcular_interseccion(linea1, linea2):
    """Calcula el punto de intersección (x, y) de dos líneas usando determinantes."""
    x1, y1, x2, y2 = linea1
    x3, y3, x4, y4 = linea2
    
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if D == 0:
        return None # Son paralelas
    
    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / D
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / D
    
    return (int(px), int(py))

print("--- LABORATORIO DE GEOMETRÍA: YOLO + HOUGHLINESP ---")

# 1. RUTAS (Ajusta la ruta de la imagen a la foto que elijas probar)
path_modelo = r"C:\Users\ivang\Desktop\Tesis_Hockey\models\best_v3.pt"
path_imagen = r"C:\Users\ivang\Downloads\FH_canchaOK.png" # <-- CAMBIA ESTO

# Cargar modelo e imagen
model = YOLO(path_modelo)
img = cv2.imread(path_imagen)

if img is None:
    print("❌ ERROR: No se pudo cargar la imagen.")
    exit()

# Redimensionar para estandarizar (igual que en el video)
img = cv2.resize(img, (800, 600))
img_resultado = img.copy()

# 2. INFERENCIA YOLO
# Usamos conf=0.3 para asegurar que encuentre la caja de la línea
results = model.predict(img, conf=0.3, verbose=False)

lineas_detectadas = [] 
for box in results[0].boxes:
    clase = int(box.cls[0])
    nombre_clase = model.names[clase].lower()
    
    # Solo nos interesan las líneas para este laboratorio
    if "line" in nombre_clase:
        # Coordenadas de la caja de YOLO
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Dibujar la caja de YOLO en AZUL (El "dónde mirar")
        cv2.rectangle(img_resultado, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_resultado, f"YOLO: {nombre_clase}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # ---------------------------------------------------------
        # 3. EL PIVOTE GEOMÉTRICO: REGRESIÓN LINEAL (cv2.fitLine)
        # ---------------------------------------------------------
        roi = img[y1:y2, x1:x2]
        if roi.size == 0: continue

       # A. Aislar solo los bordes (ignora masas de color sólido como el arco)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bordes = cv2.Canny(gray, 50, 150) # Canny es inmune al "peso" del color amarillo
        
        # B. Obtener las coordenadas (x,y) de los píxeles de los bordes
        puntos_blancos = cv2.findNonZero(bordes)

        # C. Si hay suficientes píxeles blancos, trazamos la línea matemática
        if puntos_blancos is not None and len(puntos_blancos) > 50:
            # fitLine aplica Mínimos Cuadrados para encontrar la recta
            # Devuelve el vector director (vx, vy) y un punto perteneciente a la recta (x0, y0)
            [vx, vy, x0, y0] = cv2.fitLine(puntos_blancos, cv2.DIST_L2, 0, 0.01, 0.01)
            
            vx, vy = vx[0], vy[0]
            x0, y0 = x0[0], y0[0]

            if abs(vx) > 0.001: # Evitamos errores matemáticos si la línea es 100% vertical
                m = vy / vx # Calculamos la pendiente (m)
                
                # Proyectamos la recta desde el borde izquierdo (x=0) hasta el derecho (x=ancho_caja)
                w_caja = x2 - x1
                y_izq = int(m * (0 - x0) + y0)
                y_der = int(m * (w_caja - x0) + y0)

                # D. Dibujamos la línea roja cruzando TODA la caja original
                pt1 = (x1, y1 + y_izq)
                pt2 = (x2, y1 + y_der)
                cv2.line(img_resultado, pt1, pt2, (0, 0, 255), 3)
                # Guardamos las coordenadas de la línea para calcular la intersección después
                lineas_detectadas.append((pt1[0], pt1[1], pt2[0], pt2[1]))
# ---------------------------------------------------------
# 4. ENCONTRAR EL VÉRTICE (Intersección)
# ---------------------------------------------------------
if len(lineas_detectadas) >= 2:
    # Tomamos las dos primeras líneas detectadas (la lateral y la de fondo)
    l1 = lineas_detectadas[0]
    l2 = lineas_detectadas[1]
    
    punto_interseccion = calcular_interseccion(l1, l2)
    
    if punto_interseccion:
        # Dibujamos un círculo verde grande en el punto de choque
        cv2.circle(img_resultado, punto_interseccion, 8, (0, 255, 0), -1)
        cv2.putText(img_resultado, f"Vertice: {punto_interseccion}", 
                    (punto_interseccion[0] + 10, punto_interseccion[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# 5. MOSTRAR RESULTADOS
cv2.imshow("Bordes Internos (Canny)", bordes) # Muestra cómo la PC ve el recorte
cv2.imshow("Laboratorio: Caja Azul (YOLO) vs Linea Roja (Hough)", img_resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()