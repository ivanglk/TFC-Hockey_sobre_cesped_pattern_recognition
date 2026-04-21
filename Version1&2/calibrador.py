import cv2
import numpy as np

# --- CONFIGURACIÓN ---
# Pon la ruta exacta de tu video aquí
path_video = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\3raF-Inter_D_ 2024-VelezB_2-0SICb.mp4" 

# Lista para guardar tus clics
puntos_clic = []

def clic_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Guardar coordenadas
        puntos_clic.append([x, y])
        print(f"Punto {len(puntos_clic)}: x={x}, y={y}")
        
        # Dibujar un círculo verde donde hiciste clic
        cv2.circle(imagen_con_dibujos, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Calibrador de Tesis", imagen_con_dibujos)

# Abrir video y leer el primer frame
cap = cv2.VideoCapture(path_video)
success, frame = cap.read()
cap.release()

if not success:
    print("❌ No pude leer el video. Revisa la ruta.")
    exit()

# Redimensionar al mismo tamaño que usas en tu sistema principal
# (IMPORTANTE: Debe ser el mismo tamaño que usas en pruebaTesis.py)
frame = cv2.resize(frame, (1280, 720))
imagen_con_dibujos = frame.copy()

print("--- INSTRUCCIONES ---")
print("1. Haz clic en 4 esquinas del campo en el ORDEN SIGUIENTE:")
print("   - Superior Izquierda")
print("   - Superior Derecha")
print("   - Inferior Derecha")
print("   - Inferior Izquierda")
print("2. Luego presiona cualquier tecla para obtener los datos.")

cv2.imshow("Calibrador de Tesis", imagen_con_dibujos)
cv2.setMouseCallback("Calibrador de Tesis", clic_mouse)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n--- COPIA ESTO EN TU CÓDIGO PRINCIPAL ---")
print(f"src_points = np.float32({puntos_clic})")
print("-----------------------------------------")