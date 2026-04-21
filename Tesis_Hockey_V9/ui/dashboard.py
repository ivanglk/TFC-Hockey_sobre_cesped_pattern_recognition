### Se encarga 100% de la visualización en pantalla, los botones y el minimapa táctico 2D agrupado.

import cv2
import numpy as np
import config

class DashboardUI:
    def __init__(self, callback_click):
        self.mostrar_anotaciones = True 
        self.cambio_de_lado = False 
        self.callback_click = callback_click
        
        cv2.namedWindow("Dashboard Analitico V8.1")
        cv2.setMouseCallback("Dashboard Analitico V8.1", self._click_mouse)

    def _click_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if (config.VIDEO_H - 50) <= y <= config.VIDEO_H:
                accion = None
                if 10 <= x <= 70:    accion = "SALTAR_ATRAS_5"
                elif 80 <= x <= 140: accion = "SALTAR_ATRAS_1"
                elif 150 <= x <= 230: accion = "PLAY_PAUSE"
                elif 240 <= x <= 300: accion = "SALTAR_ADELANTE_1"
                elif 310 <= x <= 370: accion = "SALTAR_ADELANTE_5"
                elif 380 <= x <= 560: 
                    self.mostrar_anotaciones = not self.mostrar_anotaciones
                elif 570 <= x <= 790: 
                    self.cambio_de_lado = not self.cambio_de_lado
                    accion = "RESETEAR_EVIDENCIA"
                
                if accion: self.callback_click(accion)

    def _dibujar_minimapa(self, lienzo, jugadores_info, centro_disputa, zona_actual, invertido):
        field_x1, field_y1 = config.VIDEO_W + 20, int(config.VIDEO_H * 0.05)
        field_w, field_h = config.PANEL_W - 40, int(config.VIDEO_H * 0.9)
        field_x2, field_y2 = field_x1 + field_w, field_y1 + field_h

        cv2.rectangle(lienzo, (config.VIDEO_W, 0), (config.TOTAL_W, config.VIDEO_H), (45, 115, 45), -1)
        
        field_border_color = (255, 255, 255)
        cv2.rectangle(lienzo, (field_x1, field_y1), (field_x2, field_y2), field_border_color, 2)
        
        cy_y = field_y1 + int(field_h * 0.5)
        cv2.line(lienzo, (field_x1, cy_y), (field_x2, cy_y), field_border_color, 2)
        cv2.circle(lienzo, (field_x1 + int(field_w*0.5), cy_y), int(field_w * 0.05), field_border_color, 1) 

        cv2.line(lienzo, (field_x1, field_y1 + int(field_h * 0.25)), (field_x2, field_y1 + int(field_h * 0.25)), (210, 210, 210), 1)
        cv2.line(lienzo, (field_x1, field_y1 + int(field_h * 0.75)), (field_x2, field_y1 + int(field_h * 0.75)), (210, 210, 210), 1)

        goal_circle_color = (250, 250, 250)
        goal_w = int(field_w * 0.6)
        goal_h = int(field_h * 0.1)
        
        gx1_vis = field_x1 + int(field_w*0.5 - goal_w*0.5)
        cv2.ellipse(lienzo, (field_x1 + int(field_w*0.5), field_y1), (goal_w // 2, goal_h), 0, 0, 180, goal_circle_color, 1) 
        cv2.rectangle(lienzo, (gx1_vis, field_y1), (gx1_vis + goal_w, field_y1 - 10), field_border_color, -1) 
        
        cv2.ellipse(lienzo, (field_x1 + int(field_w*0.5), field_y2), (goal_w // 2, int(goal_h*1.5)), 0, 180, 360, goal_circle_color, 1)
        cv2.rectangle(lienzo, (gx1_vis, field_y2), (gx1_vis + goal_w, field_y2 + 10), field_border_color, -1)

        cv2.putText(lienzo, "PIZARRA TACTICA 2D", (config.VIDEO_W + 15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cam_x, cam_y = field_x2 - 15, field_y1 + 15
        cv2.rectangle(lienzo, (cam_x - 10, cam_y - 5), (cam_x + 10, cam_y + 5), (200, 200, 200), 1) 
        cv2.rectangle(lienzo, (cam_x + 2, cam_y - 8), (cam_x + 8, cam_y + 8), (200, 200, 200), -1) 
        cv2.putText(lienzo, "HighBehind", (field_x2 - 80, field_y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        lbl_z4, lbl_z3 = ("Z1", "Z2") if invertido else ("Z4", "Z3")
        lbl_z2, lbl_z1 = ("Z3", "Z4") if invertido else ("Z2", "Z1")

        cv2.putText(lienzo, lbl_z4, (field_x1 - 25, field_y1 + int(field_h * 0.12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(lienzo, lbl_z3, (field_x1 - 25, field_y1 + int(field_h * 0.37)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(lienzo, lbl_z2, (field_x1 - 25, field_y1 + int(field_h * 0.62)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(lienzo, lbl_z1, (field_x1 - 25, field_y1 + int(field_h * 0.87)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.circle(lienzo, (field_x1, config.VIDEO_H - 15), 4, (255, 255, 255), -1) 
        cv2.putText(lienzo, "Claro", (field_x1 + 10, config.VIDEO_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.circle(lienzo, (field_x1 + 70, config.VIDEO_H - 15), 4, (0, 0, 0), -1) 
        cv2.putText(lienzo, "Oscuro", (field_x1 + 80, config.VIDEO_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        def proyectar_con_perspectiva(px, py):
            pct_x = px / config.VIDEO_W
            pct_y = py / config.VIDEO_H
            map_px = field_x1 + int(pct_x * field_w)
            map_py = field_y1 + int(pct_y * field_h)
            return max(field_x1, min(field_x2, map_px)), max(field_y1, min(field_y2, map_py))

        for (x, y, team_color) in jugadores_info:
            mx, my = proyectar_con_perspectiva(x, y)
            cv2.circle(lienzo, (mx, my), 4, team_color, -1) 
            
        if centro_disputa is not None:
            dx, dy = proyectar_con_perspectiva(centro_disputa[0], centro_disputa[1])
            cv2.circle(lienzo, (dx, dy), 12, (0, 165, 255), 1) 
            cv2.circle(lienzo, (dx, dy), 5, (0, 165, 255), -1) 

    def renderizar(self, frame_resized, annotated_frame, estado_reproduccion, tactica, cinematica, evento_trigger, coords_jugadores, centro_disputa, zona_actual_det):
        display_frame = np.zeros((config.VIDEO_H, config.TOTAL_W, 3), dtype=np.uint8)
        
        display_frame[0:config.VIDEO_H, 0:config.VIDEO_W] = annotated_frame if self.mostrar_anotaciones else frame_resized
        self._dibujar_minimapa(display_frame, coords_jugadores, centro_disputa, tactica.ultima_zona_valida, self.cambio_de_lado)

        cv2.rectangle(display_frame, (0, 0), (config.VIDEO_W, 90), (15, 15, 15), -1)
        cv2.putText(display_frame, f"CINEMATICA: {tactica.estado_posesion} (Evid: {cinematica.evidencia_posesion:.1f})", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        texto_zona = f"MEMORIA ZONA: {tactica.ultima_zona_valida}" if zona_actual_det == "Zona_Transicion" else f"VIENDO ZONA: {zona_actual_det}"
        color_zona = (0, 165, 255) if zona_actual_det == "Zona_Transicion" else (255, 255, 0)
        cv2.putText(display_frame, texto_zona, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_zona, 2)
        
        if evento_trigger:
            cv2.putText(display_frame, evento_trigger, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # TABLERO DE MÉTRICAS AGRUPADO
        cv2.putText(display_frame, "RECUPERACIONES ACUMULADAS", (config.VIDEO_W - 320, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        m_loc = tactica.metricas_recuperacion["Local"]
        m_vis = tactica.metricas_recuperacion["Visita"]

        cv2.putText(display_frame, "LOCAL", (config.VIDEO_W - 320, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display_frame, f"Def: Z1:{m_loc['Z1_ArcoLocal_25yd']} Z2:{m_loc['Z2_25yd_50yd_Local']}", (config.VIDEO_W - 320, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(display_frame, f"Ata: Z3:{m_loc['Z3_50yd_25yd_Visita']} Z4:{m_loc['Z4_25yd_ArcoVisita']}", (config.VIDEO_W - 320, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.putText(display_frame, "VISITA", (config.VIDEO_W - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(display_frame, f"Def: Z4:{m_vis['Z4_25yd_ArcoVisita']} Z3:{m_vis['Z3_50yd_25yd_Visita']}", (config.VIDEO_W - 160, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(display_frame, f"Ata: Z2:{m_vis['Z2_25yd_50yd_Local']} Z1:{m_vis['Z1_ArcoLocal_25yd']}", (config.VIDEO_W - 160, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.rectangle(display_frame, (0, config.VIDEO_H - 50), (config.VIDEO_W, config.VIDEO_H), (40, 40, 40), -1)
        
        botones = [("<< 5s", 10, 70), ("< 1s", 80, 140), (estado_reproduccion, 150, 230), ("> 1s", 240, 300), (">> 5s", 310, 370)]
        for texto, x1, x2 in botones:
            color = (0, 150, 0) if texto == "PLAY" else (0, 0, 150) if texto == "PAUSE" else (100, 100, 100)
            cv2.rectangle(display_frame, (x1, config.VIDEO_H - 40), (x2, config.VIDEO_H - 10), color, -1)
            cv2.putText(display_frame, texto, (x1 + 5, config.VIDEO_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        bx1, bx2 = 380, 560
        b_color = (130, 130, 30) if self.mostrar_anotaciones else (60, 60, 60)
        cv2.rectangle(display_frame, (bx1, config.VIDEO_H - 40), (bx2, config.VIDEO_H - 10), b_color, -1)
        cv2.putText(display_frame, "ANOTAC: " + ("ON" if self.mostrar_anotaciones else "OFF"), (bx1 + 5, config.VIDEO_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cx1, cx2 = 570, 790
        c_color = (130, 30, 130) if self.cambio_de_lado else (60, 60, 60)
        cv2.rectangle(display_frame, (cx1, config.VIDEO_H - 40), (cx2, config.VIDEO_H - 10), c_color, -1)
        cv2.putText(display_frame, "CAMBIO LADO: " + ("ON" if self.cambio_de_lado else "OFF"), (cx1 + 5, config.VIDEO_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Dashboard Analitico V8.1", display_frame)

        