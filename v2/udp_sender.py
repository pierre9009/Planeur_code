#!/usr/bin/env python3
"""
Émetteur UDP de quaternions - Lance main_fusion.py en subprocess
Usage: python udp_sender.py [IP_DEST] [PORT]
       IP par défaut: 255.255.255.255 (broadcast)
       Port par défaut: 9999
"""

import sys
import json
import struct
import socket
import subprocess

def main():
    # Config
    ip = sys.argv[1] if len(sys.argv) > 1 else "255.255.255.255"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 9999

    # Socket UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    print(f"UDP sender -> {ip}:{port}")

    # Lance main_fusion.py en subprocess
    proc = subprocess.Popen(
        [sys.executable, "main_fusion.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    # Attend "READY" sur stderr
    for line in proc.stderr:
        if "READY" in line:
            print("Fusion ready, streaming...")
            break

    # Lit JSON depuis stdout, envoie UDP binaire
    try:
        for line in proc.stdout:
            try:
                d = json.loads(line.strip())
                # Pack 4 floats little-endian: w, x, y, z
                data = struct.pack("<4f", d["qw"], d["qx"], d["qy"], d["qz"])
                sock.sendto(data, (ip, port))
            except (json.JSONDecodeError, KeyError):
                continue
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        proc.terminate()
        sock.close()

if __name__ == "__main__":
    main()
