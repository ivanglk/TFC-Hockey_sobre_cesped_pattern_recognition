### Contiene la lógica de inferencia de zonas, densidades, el cronómetro y la exportación a CSV.

import numpy as np
import pandas as pd

class MotorTactico:
    def __init__(self):
        self.metricas_recuperacion = {
            "Local": {"Z1_ArcoLocal_25yd": 0, "Z2_25yd_50yd_Local": 0, "Z3_50yd_25yd_Visita": 0, "Z4_25yd_ArcoVisita": 0},
            "Visita": {"Z1_ArcoLocal_25yd": 0, "Z2_25yd_50yd_Local": 0, "Z3_50yd_25yd_Visita": 0, "Z4_25yd_ArcoVisita": 0}
        }
        self.registro_eventos = [] 
        self.estado_posesion = "Indefinido" 
        self.ultima_zona_valida = "Z2_25yd_50yd_Local" 

    def inferir_zona_semantica(self, objetos_yolo, alto_pantalla, invertido):
        centro_accion = alto_pantalla / 2
        z_abajo = "Z4_25yd_ArcoVisita" if invertido else "Z1_ArcoLocal_25yd"
        z_arriba = "Z1_ArcoLocal_25yd" if invertido else "Z4_25yd_ArcoVisita"
        z_m_abajo = "Z3_50yd_25yd_Visita" if invertido else "Z2_25yd_50yd_Local"
        z_m_arriba = "Z2_25yd_50yd_Local" if invertido else "Z3_50yd_25yd_Visita"
        
        for obj in objetos_yolo:
            if obj["clase"] == "goal": return z_abajo if obj["cy"] > centro_accion else z_arriba
        for obj in objetos_yolo:
            if obj["clase"] == "50yd line": return z_m_arriba if obj["cy"] < centro_accion else z_m_abajo
        for obj in objetos_yolo:
            if obj["clase"] == "25yd line":
                if self.ultima_zona_valida in [z_abajo, z_m_abajo]: return z_abajo if obj["cy"] > centro_accion else z_m_abajo
                else: return z_m_arriba if obj["cy"] > centro_accion else z_arriba
        return "Zona_Transicion"

    def inferir_zona_disputa(self, jugadores_xy, radio=120, min_jugadores=4):
        if len(jugadores_xy) < min_jugadores: return None, 0
        max_vecinos = 0; centro_disputa = None
        for (cx, cy, color) in jugadores_xy:
            vecinos = [(nx, ny) for (nx, ny, ncol) in jugadores_xy if np.sqrt((cx - nx)**2 + (cy - ny)**2) < radio]
            if len(vecinos) > max_vecinos:
                max_vecinos = len(vecinos)
                centro_disputa = (int(np.mean([v[0] for v in vecinos])), int(np.mean([v[1] for v in vecinos])))
        return (centro_disputa, max_vecinos) if max_vecinos >= min_jugadores else (None, 0)

    def actualizar_logica(self, objetos, evidencia, f_actual, fps, invertido, config):
        zona_actual_det = self.inferir_zona_semantica(objetos, config.VIDEO_H, invertido)
        if zona_actual_det != "Zona_Transicion": 
            self.ultima_zona_valida = zona_actual_det
            
        evento_trigger = None
        nuevo_estado = self.estado_posesion

        if evidencia >= config.UMBRAL_EVIDENCIA_DINAMICO:
            nuevo_estado = "Ataca_Arriba (Local)" if invertido else "Ataca_Arriba (Visita)"
        elif evidencia <= -config.UMBRAL_EVIDENCIA_DINAMICO:
            nuevo_estado = "Ataca_Abajo (Visita)" if invertido else "Ataca_Abajo (Local)"

        hubo_cambio = False
        if nuevo_estado != self.estado_posesion:
            if self.estado_posesion != "Indefinido": 
                equipo_recup = "Local" if "Local" in nuevo_estado else "Visita"
                self.metricas_recuperacion[equipo_recup][self.ultima_zona_valida] += 1
                evento_trigger = f"RECUP. {equipo_recup.upper()} EN {self.ultima_zona_valida[:2]}"
                
                segundos_totales = int(f_actual / fps)
                minutos = segundos_totales // 60
                segundos = segundos_totales % 60
                tiempo_formateado = f"{minutos:02d}:{segundos:02d}"

                self.registro_eventos.append({
                    "Minuto_Video": tiempo_formateado,
                    "Equipo_Recuperador": equipo_recup,
                    "Zona_Recuperacion": self.ultima_zona_valida,
                    "Nuevo_Estado_Ataque": nuevo_estado,
                    "Cambio_Lado_Activo": invertido,
                    "Frame_Exacto": int(f_actual)
                })
            self.estado_posesion = nuevo_estado
            hubo_cambio = True
            
        return hubo_cambio, evento_trigger, zona_actual_det

    def exportar_csv(self, ruta_salida):
        df = pd.DataFrame(self.registro_eventos)
        if not df.empty: 
            df = df[["Minuto_Video", "Equipo_Recuperador", "Zona_Recuperacion", "Nuevo_Estado_Ataque", "Cambio_Lado_Activo", "Frame_Exacto"]]
            df.to_csv(ruta_salida, index=False)
            print("\n" + "="*50)
            print("--- PROCESAMIENTO FINALIZADO CORRECTAMENTE ---")
            print("="*50)
            print(f"¡ÉXITO TÁCTICO! Se registraron {len(df)} recuperaciones.")
            print(f"EL ARCHIVO FUE CREADO EXACTAMENTE AQUÍ:\n>>> {ruta_salida} <<<")
            print("-" * 50)
            print(df.head(10))
        else:
            print("ATENCIÓN: El video terminó o se presionó 'q', pero NO SE REGISTRARON CAMBIOS DE POSESIÓN.")
            