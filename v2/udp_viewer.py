#!/usr/bin/env python3
"""
Viewer OpenGL pour quaternions IMU - Affiche un planeur 3D
Usage: python udp_viewer.py [PORT]
       Port par défaut: 9999
"""

import sys
import struct
import socket
import threading
import time

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Config
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9999


class UdpReceiver(threading.Thread):
    """Thread dédié à la réception UDP - toujours le dernier quaternion"""

    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port
        self.quat = [1.0, 0.0, 0.0, 0.0]
        self.lock = threading.Lock()
        self.running = True
        self.packet_count = 0
        self.last_count_time = time.time()
        self.hz = 0

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", self.port))
        sock.settimeout(0.1)  # Timeout pour pouvoir arrêter proprement

        print(f"UDP receiver listening on port {self.port}")

        while self.running:
            try:
                data, _ = sock.recvfrom(16)
                if len(data) == 16:
                    q = struct.unpack("<4f", data)
                    with self.lock:
                        self.quat = list(q)
                        self.packet_count += 1

                    # Calcul Hz toutes les secondes
                    now = time.time()
                    if now - self.last_count_time >= 1.0:
                        with self.lock:
                            self.hz = self.packet_count
                            self.packet_count = 0
                        self.last_count_time = now

            except socket.timeout:
                continue
            except Exception as e:
                print(f"UDP error: {e}")

        sock.close()

    def get_quat(self):
        with self.lock:
            return self.quat.copy()

    def get_hz(self):
        with self.lock:
            return self.hz

    def stop(self):
        self.running = False


def draw_glider():
    """Dessine un planeur 3D simple"""

    WHITE = (0.95, 0.95, 0.95)
    BLUE = (0.2, 0.4, 0.8)
    RED = (0.8, 0.2, 0.2)
    DARK = (0.3, 0.3, 0.3)

    # === FUSELAGE ===
    glColor3f(*WHITE)
    glBegin(GL_QUADS)
    fx1, fx2 = -1.5, 1.0
    fy, fz = 0.12, 0.12
    # Dessus
    glVertex3f(fx1, fy, -fz); glVertex3f(fx1, fy, fz)
    glVertex3f(fx2, fy, fz); glVertex3f(fx2, fy, -fz)
    # Dessous
    glVertex3f(fx1, -fy, -fz); glVertex3f(fx2, -fy, -fz)
    glVertex3f(fx2, -fy, fz); glVertex3f(fx1, -fy, fz)
    # Côté gauche
    glVertex3f(fx1, -fy, -fz); glVertex3f(fx1, fy, -fz)
    glVertex3f(fx2, fy, -fz); glVertex3f(fx2, -fy, -fz)
    # Côté droit
    glVertex3f(fx1, -fy, fz); glVertex3f(fx2, -fy, fz)
    glVertex3f(fx2, fy, fz); glVertex3f(fx1, fy, fz)
    glEnd()

    # Nez
    glBegin(GL_TRIANGLES)
    nx = 1.4
    glVertex3f(fx2, fy, -fz); glVertex3f(fx2, fy, fz); glVertex3f(nx, 0, 0)
    glVertex3f(fx2, -fy, fz); glVertex3f(fx2, -fy, -fz); glVertex3f(nx, 0, 0)
    glVertex3f(fx2, fy, -fz); glVertex3f(nx, 0, 0); glVertex3f(fx2, -fy, -fz)
    glVertex3f(fx2, fy, fz); glVertex3f(fx2, -fy, fz); glVertex3f(nx, 0, 0)
    glEnd()

    # === AILES ===
    glColor3f(*BLUE)
    glBegin(GL_QUADS)
    wy = 0.03
    glVertex3f(0.3, wy, -0.12); glVertex3f(-0.1, wy, -2.5)
    glVertex3f(-0.1, wy, 2.5); glVertex3f(0.3, wy, 0.12)
    glVertex3f(0.3, -wy, -0.12); glVertex3f(0.3, -wy, 0.12)
    glVertex3f(-0.1, -wy, 2.5); glVertex3f(-0.1, -wy, -2.5)
    glEnd()

    # Bord de fuite
    glBegin(GL_QUADS)
    glVertex3f(-0.1, wy, -2.5); glVertex3f(-0.3, 0, -2.5)
    glVertex3f(-0.3, 0, 2.5); glVertex3f(-0.1, wy, 2.5)
    glVertex3f(-0.1, -wy, 2.5); glVertex3f(-0.3, 0, 2.5)
    glVertex3f(-0.3, 0, -2.5); glVertex3f(-0.1, -wy, -2.5)
    glEnd()

    # Winglets
    glColor3f(*DARK)
    glBegin(GL_TRIANGLES)
    glVertex3f(-0.1, 0, -2.5); glVertex3f(-0.3, 0, -2.5); glVertex3f(-0.2, 0.3, -2.4)
    glVertex3f(-0.1, 0, 2.5); glVertex3f(-0.2, 0.3, 2.4); glVertex3f(-0.3, 0, 2.5)
    glEnd()

    # === EMPENNAGE ===
    glColor3f(*BLUE)
    glBegin(GL_QUADS)
    hx1, hx2 = -1.2, -1.5
    hz = 0.6
    hy = 0.02
    glVertex3f(hx1, hy, -hz); glVertex3f(hx2, hy, -hz)
    glVertex3f(hx2, hy, hz); glVertex3f(hx1, hy, hz)
    glVertex3f(hx1, -hy, hz); glVertex3f(hx2, -hy, hz)
    glVertex3f(hx2, -hy, -hz); glVertex3f(hx1, -hy, -hz)
    glEnd()

    # Dérive verticale
    glColor3f(*RED)
    glBegin(GL_TRIANGLES)
    glVertex3f(-1.1, 0.1, 0); glVertex3f(-1.5, 0.1, 0); glVertex3f(-1.4, 0.5, 0)
    glEnd()

    glBegin(GL_QUADS)
    dz = 0.02
    glVertex3f(-1.1, 0.1, dz); glVertex3f(-1.5, 0.1, dz)
    glVertex3f(-1.4, 0.5, dz); glVertex3f(-1.4, 0.5, dz)
    glVertex3f(-1.1, 0.1, -dz); glVertex3f(-1.4, 0.5, -dz)
    glVertex3f(-1.5, 0.1, -dz); glVertex3f(-1.1, 0.1, -dz)
    glEnd()


def draw_axes():
    """Dessine les axes X(rouge), Y(vert), Z(bleu)"""
    glLineWidth(2)
    glBegin(GL_LINES)
    glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(2, 0, 0)
    glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, 2, 0)
    glColor3f(0, 0, 1); glVertex3f(0, 0, 0); glVertex3f(0, 0, 2)
    glEnd()


def quaternion_to_matrix(w, x, y, z):
    """Convertit quaternion en matrice OpenGL 4x4"""
    return [
        1 - 2*y*y - 2*z*z,  2*x*y - 2*w*z,      2*x*z + 2*w*y,      0,
        2*x*y + 2*w*z,      1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x,      0,
        2*x*z - 2*w*y,      2*y*z + 2*w*x,      1 - 2*x*x - 2*y*y,  0,
        0,                  0,                  0,                  1
    ]


def main():
    # Démarre le thread de réception UDP
    receiver = UdpReceiver(PORT)
    receiver.start()

    # Init PyGame/OpenGL
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("IMU Viewer - Glider")

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.15, 1)
    gluPerspective(45, display[0]/display[1], 0.1, 50.0)
    glTranslatef(0, 0, -8)

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False

        # Récupère le dernier quaternion (pas de blocage)
        quat = receiver.get_quat()
        hz = receiver.get_hz()

        # Rendu
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Axes fixes
        glPushMatrix()
        glTranslatef(-3, -2, 0)
        draw_axes()
        glPopMatrix()

        # Planeur orienté
        glPushMatrix()
        w, x, y, z = quat
        mat = quaternion_to_matrix(w, x, y, z)
        glMultMatrixf(mat)
        draw_glider()
        glPopMatrix()

        pygame.display.flip()
        pygame.display.set_caption(f"IMU Viewer - {hz} Hz")
        clock.tick(60)

    receiver.stop()
    receiver.join(timeout=1.0)
    pygame.quit()


if __name__ == "__main__":
    main()
