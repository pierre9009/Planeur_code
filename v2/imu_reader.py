#!/usr/bin/env python3
"""
Lecteur IMU simple avec validation au démarrage
Compatible avec le protocole Arduino ICM-20948
"""

import time
import struct
import serial
import math

# Protocole
SYNC1 = 0xAA
SYNC2 = 0x55

# Format du paquet Arduino (little-endian):
# uint32_t seq, float ax,ay,az, float gx,gy,gz, float mx,my,mz, float tempC, uint16_t crc
PACKET_FMT = "<I10fH"
PACKET_SIZE = struct.calcsize(PACKET_FMT)  # 4 + 40 + 2 = 46 bytes


def crc16_ccitt(data: bytes) -> int:
    """CRC-16-CCITT identique à l'Arduino"""
    crc = 0xFFFF
    for b in data:
        crc ^= (b << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class ImuReader:
    def __init__(self, port: str = "/dev/ttyS0", baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.buf = bytearray()
    
    def open(self):
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=0.1
        )
        self.ser.reset_input_buffer()
        time.sleep(0.1)
    
    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
    
    def read(self, timeout: float = 1.0):
        """Lit une mesure IMU. Retourne dict ou None si timeout."""
        t0 = time.time()
        
        while time.time() - t0 < timeout:
            # Lire données disponibles
            if self.ser.in_waiting > 0:
                self.buf.extend(self.ser.read(self.ser.in_waiting))
            
            # Chercher sync
            while True:
                idx = self.buf.find(bytes([SYNC1, SYNC2]))
                if idx < 0:
                    if len(self.buf) > 1:
                        self.buf = self.buf[-1:]
                    break
                
                if idx > 0:
                    del self.buf[:idx]
                
                # Paquet complet ?
                if len(self.buf) < 2 + PACKET_SIZE:
                    break
                
                payload = bytes(self.buf[2:2 + PACKET_SIZE])
                del self.buf[:2 + PACKET_SIZE]
                
                # Vérifier CRC
                rx_crc = struct.unpack_from("<H", payload, -2)[0]
                calc_crc = crc16_ccitt(payload[:-2])
                
                if rx_crc != calc_crc:
                    continue
                
                # Décoder
                seq, ax, ay, az, gx, gy, gz, mx, my, mz, tempC, _ = struct.unpack(PACKET_FMT, payload)
                
                return {
                    "seq": seq,
                    "ax": ax, "ay": ay, "az": az,
                    "gx": gx, "gy": gy, "gz": gz,
                    "mx": mx, "my": my, "mz": mz,
                    "tempC": tempC
                }
            
            time.sleep(0.001)
        
        return None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()


def validate_imu(port: str = "/dev/ttyS0", duration: float = 5.0):
    """
    Teste l'IMU pendant `duration` secondes.
    Vérifie:
      - Norme accéléromètre ≈ 9.81 m/s² (±1.0)
      - Norme gyroscope < 0.1 rad/s
      - Norme magnétomètre entre 30 et 70 µT
    
    Raise RuntimeError si les tests échouent.
    """
    print(f"=== Validation IMU ({duration}s) ===")
    
    acc_norms = []
    gyr_norms = []
    mag_norms = []
    
    with ImuReader(port) as imu:
        t0 = time.time()
        count = 0
        
        while time.time() - t0 < duration:
            m = imu.read(timeout=0.5)
            if m is None:
                continue
            
            count += 1
            acc_norms.append(math.sqrt(m["ax"]**2 + m["ay"]**2 + m["az"]**2))
            gyr_norms.append(math.sqrt(m["gx"]**2 + m["gy"]**2 + m["gz"]**2))
            mag_norms.append(math.sqrt(m["mx"]**2 + m["my"]**2 + m["mz"]**2))
    
    if count == 0:
        raise RuntimeError("Aucune mesure reçue - vérifier la connexion")
    
    # Moyennes
    acc_mean = sum(acc_norms) / len(acc_norms)
    gyr_mean = sum(gyr_norms) / len(gyr_norms)
    mag_mean = sum(mag_norms) / len(mag_norms)
    
    errors = []
    
    if not (8.81 < acc_mean < 10.81):
        errors.append(f"Accéléromètre hors plage: {acc_mean:.3f} m/s² (attendu 9.81 ±1.0)")
    
    if gyr_mean >= 0.1:
        errors.append(f"Gyroscope trop élevé: {gyr_mean:.4f} rad/s (attendu <0.1)")
    
    if not (30 < mag_mean < 70):
        errors.append(f"Magnétomètre hors plage: {mag_mean:.1f} µT (attendu 30-70)")
    
    if errors:
        raise RuntimeError("Validation IMU échouée:\n  - " + "\n  - ".join(errors))
    
    print("=== Validation OK ===")
    return True


# Test au chargement du module
validate_imu()


if __name__ == "__main__":
    print("\nLecture continue (Ctrl+C pour arrêter)...")
    
    with ImuReader() as imu:
        while True:
            m = imu.read()
            if m:
                print(f"seq={m['seq']:6d}  "
                      f"acc=({m['ax']:+6.2f}, {m['ay']:+6.2f}, {m['az']:+6.2f})  "
                      f"gyr=({m['gx']:+5.3f}, {m['gy']:+5.3f}, {m['gz']:+5.3f})  "
                      f"mag=({m['mx']:+6.1f}, {m['my']:+6.1f}, {m['mz']:+6.1f})")