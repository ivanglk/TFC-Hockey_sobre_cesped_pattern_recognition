### El Hilo Orquestador principal. 
# Solo importa las clases anteriores y crea el bucle del video. 
# Ejecuta este archivo para iniciar tu programa.
###

### Fíjate en la sección ejecutar(self). Ahora tiene un bloque else con un time.sleep(0.03) que actúa como un "sedante"
#   para el procesador cuando el video está en PAUSA, manteniendo la ventana viva sin colgarse.

# Archivo: main.py
import cv2
import time
import config
from core.video_reader import LectorVideoEnHilos
from core.cinematica import AnalizadorCinematico
from core.detector_yolo import DetectorYOLO
from core.motor_tactico import MotorTactico
from ui.dashboard import DashboardUI

print("--- INICIANDO SISTEMA V9.0 (MULTITHREADING ANTI-DEADLOCK) ---")

class SistemaAnalisisHockey:
    def __init__(self):
        self.lector = LectorVideoEnHilos(config.PATH_VIDEO).iniciar()
        self.fps_video = self.lector.fps
        
        self.estado_reproduccion = "PLAY"
        self.salto_solicitado = True
        self.evento_trigger = None
        self.f_actual_procesado = 0 
        
        self.cinematica = AnalizadorCinematico()
        self.yolo = DetectorYOLO(config.PATH_MODELO)
        self.tactica = MotorTactico()
        self.ui = DashboardUI(callback_click=self.procesar_click_ui)

    def procesar_click_ui(self, accion):
        f_target = self.f_actual_procesado
        
        if accion == "SALTAR_ATRAS_5":
            f_target = max(0, self.f_actual_procesado - 5 * self.fps_video)
            self.lector.saltar_a_frame(f_target)
            self.salto_solicitado = True
        elif accion == "SALTAR_ATRAS_1":
            f_target = max(0, self.f_actual_procesado - 1 * self.fps_video)
            self.lector.saltar_a_frame(f_target)
            self.salto_solicitado = True
        elif accion == "PLAY_PAUSE":
            self.estado_reproduccion = "PAUSE" if self.estado_reproduccion == "PLAY" else "PLAY"
        elif accion == "SALTAR_ADELANTE_1":
            f_target = self.f_actual_procesado + 1 * self.fps_video
            self.lector.saltar_a_frame(f_target)
            self.salto_solicitado = True
        elif accion == "SALTAR_ADELANTE_5":
            f_target = self.f_actual_procesado + 5 * self.fps_video
            self.lector.saltar_a_frame(f_target)
            self.salto_solicitado = True
        elif accion == "RESETEAR_EVIDENCIA":
            self.cinematica.evidencia_posesion = 0.0

    def ejecutar(self):
        annotated_frame = None

        while True:
            # 1. MODO PLAY O SALTO
            if self.estado_reproduccion == "PLAY" or self.salto_solicitado:
                
                if self.lector.hay_frames():
                    exito, frame, f_idx = self.lector.leer()
                    if not exito: break
                    
                    self.f_actual_procesado = f_idx 
                    
                    if self.salto_solicitado:
                        self.cinematica.resetear()
                        self.evento_trigger = None
                        self.salto_solicitado = False
                    
                    frame_resized = cv2.resize(frame, (config.VIDEO_W, config.VIDEO_H))
                    annotated_frame = frame_resized.copy()
                    
                    # --- A. CINEMÁTICA ---
                    evidencia = self.cinematica.actualizar(frame_resized, f_idx)
                    
                    # --- B. DETECCIÓN YOLO ---
                    objetos, coords_jugadores = self.yolo.procesar_frame(frame_resized)
                    
                    if self.ui.mostrar_anotaciones:
                        for obj in objetos:
                            color = (255,0,0) if obj["clase"] == "goal" else (0,255,0)
                            cv2.circle(annotated_frame, (obj["cx"], obj["cy"]), 8, color, -1)
                        for c in coords_jugadores:
                            cv2.circle(annotated_frame, (c[0], c[1]), 3, (0, 255, 255), -1)
                    
                    # --- C. MOTOR TÁCTICO ---
                    centro_disputa, _ = self.tactica.inferir_zona_disputa(coords_jugadores)
                    if self.ui.mostrar_anotaciones and centro_disputa is not None:
                        cv2.circle(annotated_frame, centro_disputa, 120, (0, 165, 255), 2)

                    hubo_cambio, evento, zona_det = self.tactica.actualizar_logica(
                        objetos, evidencia, f_idx, self.fps_video, self.ui.cambio_de_lado, config)
                    
                    if hubo_cambio:
                        self.evento_trigger = evento
                        self.cinematica.evidencia_posesion = 0.0
                    
                    # --- D. RENDERIZADO UI ---
                    self.ui.renderizar(
                        frame_resized, annotated_frame, self.estado_reproduccion,
                        self.tactica, self.cinematica, self.evento_trigger,
                        coords_jugadores, centro_disputa, zona_det
                    )
                else:
                    # El hilo lector está demorado, dormimos el cerebro 10 milisegundos
                    time.sleep(0.01)
            
            # 2. MODO PAUSA
            else:
                # Si estamos en pausa, relajamos la CPU para que Windows no se cuelgue
                time.sleep(0.03)

            # 3. ESCUCHA DE EVENTOS DE TECLADO Y MOUSE
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            elif k == ord(' '): 
                self.estado_reproduccion = "PAUSE" if self.estado_reproduccion == "PLAY" else "PLAY"

        self.lector.detener()
        cv2.destroyAllWindows()
        self.tactica.exportar_csv(config.RUTA_SALIDA_CSV)

if __name__ == "__main__":
    app = SistemaAnalisisHockey()
    app.ejecutar()