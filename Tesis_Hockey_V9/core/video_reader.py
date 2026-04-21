### Hilo secundario que se dedica pura y exclusivamente a leer fotogramas del disco duro lo más rapido posible
# Los mete en una queue (cola)
# Añadimos time.sleep y metimos el lector dentro de la función del hilo. 

# Archivo: core/video_reader.py
import cv2
import threading
import queue
import time

class LectorVideoEnHilos:
    def __init__(self, path_video, tamano_cola=30):
        self.path_video = path_video
        
        # Leemos los FPS rápido y cerramos, para no trabar a Windows
        temp_cap = cv2.VideoCapture(path_video)
        self.fps = temp_cap.get(cv2.CAP_PROP_FPS) or 30
        temp_cap.release()
        
        self.cola = queue.Queue(maxsize=tamano_cola)
        self.detenido = False
        self.salto_pendiente = None 

    def iniciar(self):
        hilo = threading.Thread(target=self._actualizar, args=())
        hilo.daemon = True 
        hilo.start()
        return self

    def _actualizar(self):
        # INICIALIZAMOS EL VIDEO ADENTRO DEL HILO SECUNDARIO (¡Clave para Windows!)
        cap = cv2.VideoCapture(self.path_video)
        
        while not self.detenido:
            if self.salto_pendiente is not None:
                with self.cola.mutex:
                    self.cola.queue.clear() 
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.salto_pendiente)
                self.salto_pendiente = None

            if not self.cola.full():
                f_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
                exito, frame = cap.read()
                
                if not exito:
                    self.detenido = True
                    break
                
                self.cola.put((exito, frame, f_idx))
            else:
                # Si la bandeja está llena, el hilo "duerme" 10 milisegundos para no quemar el CPU
                time.sleep(0.01)
                
        cap.release()

    def leer(self):
        return self.cola.get()

    def hay_frames(self):
        return not self.cola.empty()

    def saltar_a_frame(self, target_frame):
        self.salto_pendiente = target_frame

    def detener(self):
        self.detenido = True