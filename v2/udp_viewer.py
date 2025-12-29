#!/usr/bin/env python3
"""
Viewer OpenGL pour quaternions IMU - Affiche un planeur 3D
Usage: python udp_viewer.py [PORT]
       Port par défaut: 9999
"""

import sys
import struct
import socket
import time

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

# Config
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9999


def draw_glider():
    """Dessine un planeur 3D simple"""

    # Couleurs
    WHITE = (0.95, 0.95, 0.95)
    BLUE = (0.2, 0.4, 0.8)
    RED = (0.8, 0.2, 0.2)
    DARK = (0.3, 0.3, 0.3)

    # === FUSELAGE (corps principal) ===
    glColor3f(*WHITE)
    glBegin(GL_QUADS)
    # Fuselage: prisme de -1.5 à +1.0 sur X, section carrée 0.15
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

    # Nez (cône simplifié en triangle)
    glBegin(GL_TRIANGLES)
    nx = 1.4  # pointe du nez
    # Dessus
    glVertex3f(fx2, fy, -fz); glVertex3f(fx2, fy, fz); glVertex3f(nx, 0, 0)
    # Dessous
    glVertex3f(fx2, -fy, fz); glVertex3f(fx2, -fy, -fz); glVertex3f(nx, 0, 0)
    # Côtés
    glVertex3f(fx2, fy, -fz); glVertex3f(nx, 0, 0); glVertex3f(fx2, -fy, -fz)
    glVertex3f(fx2, fy, fz); glVertex3f(fx2, -fy, fz); glVertex3f(nx, 0, 0)
    glEnd()

    # === AILES ===
    glColor3f(*BLUE)
    glBegin(GL_QUADS)
    # Aile: de Z=-2.5 à Z=+2.5, X de 0.3 à -0.3 (légèrement en flèche)
    # Épaisseur: 0.03
    wy = 0.03
    # Dessus aile
    glVertex3f(0.3, wy, -0.12); glVertex3f(-0.1, wy, -2.5)
    glVertex3f(-0.1, wy, 2.5); glVertex3f(0.3, wy, 0.12)
    # Dessous aile
    glVertex3f(0.3, -wy, -0.12); glVertex3f(0.3, -wy, 0.12)
    glVertex3f(-0.1, -wy, 2.5); glVertex3f(-0.1, -wy, -2.5)
    glEnd()

    # Bord de fuite aile (arrière)
    glBegin(GL_QUADS)
    glVertex3f(-0.1, wy, -2.5); glVertex3f(-0.3, 0, -2.5)
    glVertex3f(-0.3, 0, 2.5); glVertex3f(-0.1, wy, 2.5)
    glVertex3f(-0.1, -wy, 2.5); glVertex3f(-0.3, 0, 2.5)
    glVertex3f(-0.3, 0, -2.5); glVertex3f(-0.1, -wy, -2.5)
    glEnd()

    # Bouts d'ailes (winglets)
    glColor3f(*DARK)
    glBegin(GL_TRIANGLES)
    # Winglet gauche
    glVertex3f(-0.1, 0, -2.5); glVertex3f(-0.3, 0, -2.5); glVertex3f(-0.2, 0.3, -2.4)
    # Winglet droit
    glVertex3f(-0.1, 0, 2.5); glVertex3f(-0.2, 0.3, 2.4); glVertex3f(-0.3, 0, 2.5)
    glEnd()

    # === EMPENNAGE (queue) ===
    # Stabilisateur horizontal
    glColor3f(*BLUE)
    glBegin(GL_QUADS)
    hx1, hx2 = -1.2, -1.5
    hz = 0.6
    hy = 0.02
    # Dessus
    glVertex3f(hx1, hy, -hz); glVertex3f(hx2, hy, -hz)
    glVertex3f(hx2, hy, hz); glVertex3f(hx1, hy, hz)
    # Dessous
    glVertex3f(hx1, -hy, hz); glVertex3f(hx2, -hy, hz)
    glVertex3f(hx2, -hy, -hz); glVertex3f(hx1, -hy, -hz)
    glEnd()

    # Dérive verticale (gouvernail)
    glColor3f(*RED)
    glBegin(GL_TRIANGLES)
    # Dérive
    glVertex3f(-1.1, 0.1, 0); glVertex3f(-1.5, 0.1, 0); glVertex3f(-1.4, 0.5, 0)
    glEnd()

    # Épaisseur dérive
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
    # X - rouge
    glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(2, 0, 0)
    # Y - vert
    glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, 2, 0)
    # Z - bleu
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
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("IMU Viewer - Glider")

    # OpenGL setup
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.1, 0.1, 0.15, 1)

    gluPerspective(45, display[0]/display[1], 0.1, 50.0)
    glTranslatef(0, 0, -8)

    # Socket UDP non-bloquant avec buffer minimal
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64)  # ~4 paquets max
    sock.bind(("", PORT))
    sock.setblocking(False)

    print(f"Listening on UDP port {PORT}...")

    # Vide le buffer au démarrage
    while True:
        try:
            sock.recvfrom(16)
        except BlockingIOError:
            break

    # État
    quat = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
    recv_times = []
    hz = 0.0

    clock = pygame.time.Clock()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False

        # Réception UDP (non-bloquant, prend le dernier paquet)
        while True:
            try:
                data, _ = sock.recvfrom(16)
                if len(data) == 16:
                    quat = list(struct.unpack("<4f", data))
                    recv_times.append(time.time())
            except BlockingIOError:
                break

        # Calcul Hz (sur dernière seconde)
        now = time.time()
        recv_times = [t for t in recv_times if now - t < 1.0]
        hz = len(recv_times)

        # Rendu
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Axes fixes (référence)
        glPushMatrix()
        glTranslatef(-3, -2, 0)
        draw_axes()
        glPopMatrix()

        # Planeur orienté par quaternion
        glPushMatrix()
        w, x, y, z = quat
        mat = quaternion_to_matrix(w, x, y, z)
        glMultMatrixf(mat)
        draw_glider()
        glPopMatrix()

        pygame.display.flip()

        # Affiche Hz dans le titre
        pygame.display.set_caption(f"IMU Viewer - {hz} Hz")

        clock.tick(60)

    sock.close()
    pygame.quit()


if __name__ == "__main__":
    main()
