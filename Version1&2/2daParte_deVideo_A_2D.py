import cv2
import numpy as np
from ultralytics import YOLO
import os

print("--- INICIANDO INTERFAZ TÁCTICA ---")

# 1. RUTAS
path_modelo = r"c:\Users\ivang\Desktop\Tesis_Hockey\models\best.pt"
path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\3raF-Inter_D_ 2024-VelezB_2-0SICb.mp4" 

# 2. CARGAR MODELO Y VIDEO
model = YOLO(path_modelo)
cap = cv2.VideoCapture(path_video)

if not cap.isOpened():
    print("❌ ERROR: OpenCV no pudo abrir el video.")
    exit()

# --- FUNCIÓN PARA DIBUJAR LA PIZARRA 2D ---
def crear_pizarra_hockey(alto=600, ancho=350):
    """Dibuja una cancha de hockey vertical estilizada usando NumPy y OpenCV"""
    # Crear fondo verde oscuro (Césped sintético)
    pizarra = np.zeros((alto, ancho, 3), dtype=np.uint8)
    pizarra[:] = (40, 110, 40) # Color BGR (verde táctico)
    
    color_linea = (255, 255, 255)
    grosor = 2

    # Medidas relativas
    medio_y = alto // 2
    linea_23_sup = int(alto * 0.25)
    linea_23_inf = int(alto * 0.75)
    radio_area = int(ancho * 0.3)
    
    # 1. Borde exterior
    cv2.rectangle(pizarra, (0, 0), (ancho-1, alto-1), color_linea, grosor)
    # 2. Línea central
    cv2.line(pizarra, (0, medio_y), (ancho, medio_y), color_linea, grosor)
    # 3. Líneas de 23 metros (aproximadas al cuarto de cancha)
    cv2.line(pizarra, (0, linea_23_sup), (ancho, linea_23_sup), color_linea, grosor)
    cv2.line(pizarra, (0, linea_23_inf), (ancho, linea_23_inf), color_linea, grosor)
    
    # 4. Áreas (Arcos) - Dibujamos medias elipses
    centro_arco_sup = (ancho // 2, 0)
    centro_arco_inf = (ancho // 2, alto)
    cv2.ellipse(pizarra, centro_arco_sup, (radio_area, radio_area), 0, 0, 180, color_linea, grosor)
    cv2.ellipse(pizarra, centro_arco_inf, (radio_area, radio_area), 0, 180, 360, color_linea, grosor)
    
    # 5. Punto de penal
    cv2.circle(pizarra, (ancho // 2, int(alto * 0.12)), 3, color_linea, -1)
    cv2.circle(pizarra, (ancho // 2, int(alto * 0.88)), 3, color_linea, -1)

    return pizarra

# --- BUCLE PRINCIPAL ---
print("🎥 Iniciando Dashboard Táctico... (Presiona 'q' para salir)")

def clasificar_equipo(frame, box):
    """
    Recorta el pecho del jugador, ignora el pasto verde y determina el equipo.
    Devuelve un color BGR para dibujar en la pizarra y el nombre del equipo.
    """
    x_centro, y_centro, w, h = box
    
    # 1. Recortar solo el "torso" (evita piernas y cabeza)
    x1 = int(x_centro - w/4)
    x2 = int(x_centro + w/4)
    y1 = int(y_centro - h/2)  # Desde arriba de la caja
    y2 = int(y_centro)        # Hasta la mitad
    
    # Evitar errores si el jugador está en el borde de la pantalla
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    
    torso = frame[y1:y2, x1:x2]
    
    # Si la caja es muy pequeña o inválida, devolver color gris genérico
    if torso.size == 0:
        return (128, 128, 128), "Desconocido"
        
    # 2. Filtrar el pasto verde usando HSV
    hsv_torso = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    
    # Rango de color verde en HSV (H: 35 a 85 aprox)
    # Hacemos una máscara que tome todo lo que NO SEA verde
    mask_no_pasto = cv2.inRange(hsv_torso, (0, 0, 0), (34, 255, 255)) | \
                    cv2.inRange(hsv_torso, (86, 0, 0), (179, 255, 255))
    
    # 3. Calcular el color promedio (BGR) solo de los píxeles de la camiseta
    color_bgr = cv2.mean(torso, mask=mask_no_pasto)[:3]
    
    # 4. Lógica de Clasificación (Ejemplo: Claro vs Oscuro)
    # La luminosidad se calcula sumando los canales con pesos estándar
    luminosidad = color_bgr[0]*0.114 + color_bgr[1]*0.587 + color_bgr[2]*0.299
    
    # AJUSTA ESTE VALOR (100) SEGÚN LO QUE VEAS EN TU VIDEO
    UMBRAL_LUMINOSIDAD = 120 
    
    if luminosidad > UMBRAL_LUMINOSIDAD:
        # Equipo Claro (ej. Blanco)
        return (255, 255, 255), "Equipo Blanco" 
    else:
        # Equipo Oscuro (ej. Azul/Negro)
        return (255, 0, 0), "Equipo Azul" # BGR para Azul puro


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 1. Redimensionar el video para que quepa bien en pantalla
    video_w, video_h = 800, 600
    frame_resized = cv2.resize(frame, (video_w, video_h))

    # 2. Inferencia de YOLO
    results = model.predict(frame_resized, conf=0.5, verbose=False)
    
    # Dibujar las cajas de YOLO en el video
    annotated_frame = results[0].plot()

    # 3. Crear una pizarra limpia en cada frame
    pizarra_w = 350
    pizarra_h = video_h
    mapa_2d = crear_pizarra_hockey(alto=pizarra_h, ancho=pizarra_w)

    # 4. MAPEO (Aquí irá la Homografía después)
    # Por ahora, extraemos el centro de cada jugador y lo "proyectamos" de forma cruda
    for box in results[0].boxes:
        clase = int(box.cls[0])
        nombre_clase = model.names[clase]
        
        if nombre_clase == "player" or nombre_clase == "Player":
            # Coordenadas de la caja (x_centro, y_centro, ancho, alto)
            x_centro, y_centro, w, h = box.xywh[0]
            
            # El punto de contacto con el suelo es el centro en X, y la base en Y
            x_suelo = int(x_centro)
            y_suelo = int(y_centro + (h / 2))
            
          
          # --- SIMULACIÓN TEMPORAL DE HOMOGRAFÍA (CON INVERSIÓN) ---
            map_x = int((x_suelo / video_w) * pizarra_w)
            
            # Al restar el cálculo a la altura total (pizarra_h), invertimos el eje Y
            map_y = pizarra_h - int((y_suelo / video_h) * pizarra_h)
            
         # --- NUEVO: CLASIFICAR EQUIPO ---
            # Le pasamos el frame original sin redimensionar (o el redimensionado, pero asegúrate 
            # de que las coordenadas de YOLO correspondan a la misma imagen). 
            # Como YOLO corrió sobre frame_resized, le pasamos frame_resized.
            color_equipo, nombre_equipo = clasificar_equipo(frame_resized, box.xywh[0])
            
            # Dibujar al jugador en la pizarra con el COLOR DE SU EQUIPO
            cv2.circle(mapa_2d, (map_x, map_y), 6, color_equipo, -1)
            cv2.circle(mapa_2d, (map_x, map_y), 6, (0, 0, 0), 1) # Borde negro

    # 5. CONSTRUIR EL DASHBOARD FINAL
    # Juntamos el video y la pizarra de lado a lado (Horizontal Stack)
    dashboard = np.hstack((annotated_frame, mapa_2d))

    # Añadir un título bonito
    cv2.rectangle(dashboard, (0, 0), (dashboard.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(dashboard, "TESIS HOCKEY - Dashboard Analitico v1.0", (20, 28), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Mostrar
    cv2.imshow("TEST PROTOTIPO - Ivan", dashboard)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()