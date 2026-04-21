### Se encarga exclusivamente de la red neuronal y extraer las cajas delimitadoras y colores de equipo.

import cv2
import numpy as np
from ultralytics import YOLO
import config

class DetectorYOLO:
    def __init__(self, path_modelo):
        self.model = YOLO(path_modelo)
        self.nombres = self.model.names

    def _obtener_metricas_caja(self, xyxy):
        x1, y1, x2, y2 = xyxy
        return (int((x1 + x2) / 2), int((y1 + y2) / 2), int(y2 - y1))

    def procesar_frame(self, frame_resized):
        results = self.model.predict(frame_resized, conf=0.15, imgsz=800, agnostic_nms=True, verbose=False)
        objetos_detectados = []
        coords_jugadores_minimapa = [] 

        cajas_arcos_loop = [box.xyxy[0].cpu().numpy() for box in results[0].boxes if self.nombres[int(box.cls[0])].lower() == "goal"]

        for box in results[0].boxes:
            nombre_clase = self.nombres[int(box.cls[0])].lower()
            xyxy = box.xyxy[0].cpu().numpy()
            cx, cy, alto_box = self._obtener_metricas_caja(xyxy)
            
            if "line" in nombre_clase:
                if any((a[0]-15) < cx < (a[2]+15) and (a[1]-15) < cy < (a[3]+15) for a in cajas_arcos_loop): continue
            
            objetos_detectados.append({"clase": nombre_clase, "cy": cy, "cx": cx, "alto_box": alto_box, "xyxy": xyxy})
            
            if "player" in nombre_clase:
                x1, y1, x2, y2 = map(int, xyxy)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(config.VIDEO_W, x2), min(config.VIDEO_H, y2)
                if x2 > x1 and y2 > y1:
                    player_hsv = cv2.cvtColor(frame_resized[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
                    team_color = (255, 255, 255) if np.mean(player_hsv[:, :, 2]) > 150 else (0, 0, 0)
                    coords_jugadores_minimapa.append((cx, cy, team_color))

        return objetos_detectados, coords_jugadores_minimapa