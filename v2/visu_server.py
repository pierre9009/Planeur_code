#!/usr/bin/env python3
"""Serveur WebSocket pour visualisation 3D - Lance main_fusion.py en subprocess"""

import asyncio
import sys
import os
import websockets
import logging

# Réduire les logs websockets (erreurs de connexion bénignes)
logging.getLogger('websockets').setLevel(logging.ERROR)

# Chemin vers main_fusion.py (même répertoire)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_SCRIPT = os.path.join(SCRIPT_DIR, "main_fusion.py")

# Fréquence d'envoi WebSocket (Hz) - plus bas = moins de latence réseau
WS_SEND_RATE = 30


class FusionBridge:
    """Pont entre le subprocess de fusion et les clients WebSocket"""

    def __init__(self):
        self.process = None
        self.clients = set()
        self.latest_data = None

    async def start_fusion(self):
        """Démarre main_fusion.py en subprocess"""
        print(f"Lancement de {FUSION_SCRIPT}")

        self.process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", FUSION_SCRIPT,  # -u = unbuffered
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Attendre READY sur stderr
        line = await self.process.stderr.readline()
        if b"READY" in line:
            print("Fusion IMU prête")
        else:
            print(f"Message fusion: {line.decode().strip()}")

        # Lancer la lecture du stdout (consomme sans bloquer)
        asyncio.create_task(self.read_fusion_output())

        # Lancer l'envoi périodique aux clients
        asyncio.create_task(self.broadcast_loop())

    async def read_fusion_output(self):
        """Lit les données JSON depuis main_fusion.py (consomme en continu)"""
        while True:
            line = await self.process.stdout.readline()
            if not line:
                print("Subprocess fusion terminé")
                break
            # Stocke uniquement la dernière valeur
            self.latest_data = line.decode().strip()

    async def broadcast_loop(self):
        """Envoie les données aux clients à fréquence fixe"""
        interval = 1.0 / WS_SEND_RATE

        while True:
            if self.latest_data and self.clients:
                # Envoyer à tous les clients en parallèle
                dead_clients = set()
                for client in self.clients:
                    try:
                        await client.send(self.latest_data)
                    except websockets.ConnectionClosed:
                        dead_clients.add(client)

                self.clients -= dead_clients

            await asyncio.sleep(interval)

    async def handle_client(self, websocket, path=None):
        """Gère une connexion client WebSocket"""
        print(f"Client connecté ({len(self.clients) + 1} clients)")
        self.clients.add(websocket)

        try:
            async for _ in websocket:
                pass
        except websockets.ConnectionClosed:
            pass
        finally:
            self.clients.discard(websocket)
            print(f"Client déconnecté ({len(self.clients)} clients)")

    def stop(self):
        """Arrête le subprocess"""
        if self.process:
            self.process.terminate()


async def main():
    bridge = FusionBridge()

    # Démarrer la fusion
    await bridge.start_fusion()

    # Démarrer le serveur WebSocket
    print(f"Serveur WebSocket sur ws://0.0.0.0:8765 ({WS_SEND_RATE}Hz)")
    async with websockets.serve(bridge.handle_client, "0.0.0.0", 8765):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nArrêt")
