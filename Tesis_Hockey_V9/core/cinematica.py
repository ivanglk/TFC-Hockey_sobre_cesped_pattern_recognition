### Encapsula el algoritmo de Lucas-Kanade y el acumulador de evidencia. 

import cv2
import numpy as np
import config

class AnalizadorCinematico:
    def __init__(self):
        self.color_gris_previo = None
        self.puntos_previos = None
        self.evidencia_posesion = 0.0

    def resetear(self):
        self.color_gris_previo = None
        self.puntos_previos = None
        self.evidencia_posesion = 0.0

    def actualizar(self, frame_resized, frame_idx):
        color_gris_actual = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
        movimiento_y = 0.0

        if self.color_gris_previo is not None and self.puntos_previos is not None and len(self.puntos_previos) > 0:
            p_act, st, err = cv2.calcOpticalFlowPyrLK(self.color_gris_previo, color_gris_actual, self.puntos_previos, None, **config.LK_PARAMS)
            if p_act is not None:
                p_nuevos = p_act[st == 1]
                p_viejos = self.puntos_previos[st == 1]
                if len(p_nuevos) > 0:
                    movimiento_y = np.mean(p_nuevos[:, 1] - p_viejos[:, 1])
                    if abs(movimiento_y) > config.UMBRAL_MOVIMIENTO: 
                        self.evidencia_posesion += (movimiento_y * 1.5) 
                    else:
                        self.evidencia_posesion *= 0.85 
                    self.evidencia_posesion = max(-25.0, min(25.0, self.evidencia_posesion))
        
        if frame_idx % 5 == 0 or self.puntos_previos is None or len(self.puntos_previos) < 10:
            self.puntos_previos = cv2.goodFeaturesToTrack(color_gris_actual, mask=None, **config.FEATURE_PARAMS)
        
        self.color_gris_previo = color_gris_actual.copy()
        return self.evidencia_posesion