#!/usr/bin/env python3
"""Serveur WebSocket pour visualisation 3D - Lance main_fusion.py en subprocess"""

import asyncio
import subprocess
import sys
import os
import websockets

# Chemin vers main_fusion.py (même répertoire)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FUSION_SCRIPT = os.path.join(SCRIPT_DIR, "main_fusion.py")


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
            sys.executable, FUSION_SCRIPT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        # Attendre READY sur stderr
        line = await self.process.stderr.readline()
        if b"READY" in line:
            print("Fusion IMU prête")
        else:
            print(f"Message fusion: {line.decode().strip()}")

        # Lancer la lecture du stdout
        asyncio.create_task(self.read_fusion_output())

    async def read_fusion_output(self):
        """Lit les données JSON depuis main_fusion.py et les diffuse"""
        while True:
            line = await self.process.stdout.readline()
            if not line:
                print("Subprocess fusion terminé")
                break

            self.latest_data = line.decode().strip()

            # Diffuser à tous les clients connectés
            if self.clients:
                await asyncio.gather(
                    *[self.send_to_client(client) for client in self.clients],
                    return_exceptions=True
                )

    async def send_to_client(self, client):
        """Envoie les dernières données à un client"""
        try:
            await client.send(self.latest_data)
        except websockets.ConnectionClosed:
            self.clients.discard(client)

    async def handle_client(self, websocket, path=None):
        """Gère une connexion client WebSocket"""
        print(f"Client connecté ({len(self.clients) + 1} clients)")
        self.clients.add(websocket)

        try:
            # Garder la connexion ouverte
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
    print("Serveur WebSocket sur ws://0.0.0.0:8765")
    async with websockets.serve(bridge.handle_client, "0.0.0.0", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nArrêt")
