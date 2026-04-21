##  input_video_path = "videos_input/3raF-Inter_D_ 2024-VélezB_2-0SICb.mp4" # Cambia por el nombre de tu video

import cv2
from ultralytics import YOLO
import os

print("--- INICIANDO DIAGNÓSTICO ---")

# 1. DEFINIR RUTAS (Usa rutas absolutas para evitar errores)
# Ajusta esto si tu usuario es diferente o el archivo se llama distinto
path_modelo = r"c:\Users\ivang\Desktop\Tesis_Hockey\models\best.pt"
path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\3raF-Inter_D_ 2024-VelezB_2-0SICb.mp4" 

# 2. VERIFICACIÓN DE ARCHIVOS (Aquí suele estar el error)
if not os.path.exists(path_modelo):
    print(f"❌ ERROR CRÍTICO: No encuentro el modelo en: {path_modelo}")
    exit()
else:
    print("✅ Modelo encontrado.")

if not os.path.exists(path_video):
    print(f"❌ ERROR CRÍTICO: No encuentro el video en: {path_video}")
    print("   -> Verifica que el nombre sea exacto (mayúsculas/minúsculas).")
    print("   -> Verifica que la extensión sea .mp4 (y no .mp4.mp4).")
    exit()
else:
    print(f"✅ Video encontrado: {path_video}")

# 3. CARGAR MODELO
print("⏳ Cargando IA (esto puede tardar unos segundos)...")
try:
    model = YOLO(path_modelo)
    print("✅ IA Cargada correctamente.")
except Exception as e:
    print(f"❌ Error al cargar YOLO: {e}")
    exit()

# 4. ABRIR VIDEO
cap = cv2.VideoCapture(path_video)

if not cap.isOpened():
    print("❌ ERROR: OpenCV no pudo abrir el video, aunque el archivo existe.")
    print("   -> Puede ser un formato no soportado (codec). Intenta con otro video.")
    exit()

print("🎥 Abriendo ventana de video... (Presiona 'q' en la ventana para salir)")

# 5. BUCLE DE PROCESAMIENTO
frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("ℹ️ Fin del video o error de lectura de frame.")
        break

    frame_count += 1
    
    # Procesar solo 1 de cada 3 frames para ir más rápido (opcional)
    # if frame_count % 3 != 0: continue 

    # Redimensionar (ayuda a ver la ventana si el video es 4K)
    frame = cv2.resize(frame, (1020, 600))

    # Inferencia
    results = model.predict(frame, conf=0.5, verbose=False) # verbose=False limpia la consola
    annotated_frame = results[0].plot()

    # Mostrar
    cv2.imshow("TEST PROTOTIPO - Ivan", annotated_frame)

    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("🛑 Usuario cerró el programa.")
        break

cap.release()
cv2.destroyAllWindows()
print("--- FIN DEL PROGRAMA ---")