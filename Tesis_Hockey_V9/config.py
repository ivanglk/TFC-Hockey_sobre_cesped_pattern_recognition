### Guarda las rutas, dimensiones y parámetros. Si mañana necesitas ajustar algo, vienes directo aquí.

import cv2

# ==========================================
# RUTAS
# ==========================================
PATH_MODELO = r"C:\Users\ivang\Desktop\Tesis_Hockey\models\best_v5.pt" 
PATH_VIDEO = r"C:\Users\ivang\Desktop\Tesis_Hockey\videos_inputs\6taFecha-2024-Vélez _B_ 2-0DAOM.mp4" 
RUTA_SALIDA_CSV = r"C:\Users\ivang\Desktop\Tesis_Hockey\Tesis_Hockey_V9\recuperaciones_hockey_final.csv"

# ==========================================
# DIMENSIONES UI
# ==========================================
VIDEO_W, VIDEO_H = 800, 600
PANEL_W = 280  
TOTAL_W = VIDEO_W + PANEL_W

# ==========================================
# PARÁMETROS ALGORÍTMICOS
# ==========================================
UMBRAL_MOVIMIENTO = 0.05     
UMBRAL_EVIDENCIA_DINAMICO = 6 

LK_PARAMS = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
FEATURE_PARAMS = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)