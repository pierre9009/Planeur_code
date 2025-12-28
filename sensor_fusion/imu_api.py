#!/usr/bin/env python3
"""
IMU API utilisant l'UART matériel du Raspberry Pi (/dev/serial0 ou /dev/ttyAMA0)
Beaucoup plus fiable que le bit-banging avec pigpio
"""

import time
import struct
import serial

SYNC1 = 0xAA
SYNC2 = 0x55
FMT = "<I10fH"
PKT_SIZE = struct.calcsize(FMT)

def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """Calcul CRC-16-CCITT"""
    crc = init
    for b in data:
        crc ^= (b << 8) & 0xFFFF
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


class ImuHardwareUart:

    
    def __init__(self, port: str = "/dev/ttyS0", baudrate: int = 115200):

        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.buf = bytearray()
        self.bad_crc = 0
        self.total = 0
        self.timeouts = 0
    
    def open(self):
        """Ouvre la connexion série"""
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,  # Timeout lecture
                write_timeout=1.0
            )
            
            # Flush les buffers
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            
            # Attendre un peu que l'Arduino se stabilise
            time.sleep(0.1)
            
            print(f"✓ UART ouvert: {self.port} @ {self.baudrate} baud")
            
        except serial.SerialException as e:
            raise RuntimeError(f"Impossible d'ouvrir {self.port}: {e}")
    
    def close(self):
        """Ferme la connexion série"""
        if self.serial is not None and self.serial.is_open:
            self.serial.close()
            self.serial = None
            print(f"✓ UART fermé")
    
    def _feed(self):
        """Lit les données disponibles du port série"""
        if self.serial.in_waiting > 0:
            data = self.serial.read(self.serial.in_waiting)
            self.buf.extend(data)
    
    def read_measurement(self, timeout_s: float = 1.0):
        """
        Lit une mesure IMU complète
        
        Args:
            timeout_s: Timeout en secondes
            
        Returns:
            dict avec les données IMU ou None si timeout
        """
        t0 = time.perf_counter()
        
        while True:
            # Lire les données disponibles
            self._feed()
            
            # Chercher et traiter les paquets dans le buffer
            while True:
                # Chercher les octets de synchronisation
                idx = self.buf.find(bytes([SYNC1, SYNC2]))
                
                if idx < 0:
                    # Pas de sync trouvé, garder le dernier octet au cas où
                    if len(self.buf) > 1:
                        self.buf[:] = self.buf[-1:]
                    break
                
                # Supprimer les données avant le sync
                if idx > 0:
                    del self.buf[:idx]
                
                # Vérifier si on a assez de données pour un paquet complet
                if len(self.buf) < 2 + PKT_SIZE:
                    break
                
                # Extraire le payload (sans SYNC1 et SYNC2)
                payload = bytes(self.buf[2:2 + PKT_SIZE])
                del self.buf[:2 + PKT_SIZE]
                
                self.total += 1
                
                # Vérifier le CRC
                rx_crc = struct.unpack_from("<H", payload, PKT_SIZE - 2)[0]
                calc_crc = crc16_ccitt(payload[:-2])
                
                if rx_crc != calc_crc:
                    self.bad_crc += 1
                    continue  # Paquet invalide, chercher le suivant
                
                # Décoder le paquet
                seq, ax, ay, az, gx, gy, gz, mx, my, mz, tempC, _ = struct.unpack(FMT, payload)
                
                return {
                    "seq": int(seq),
                    "ax": float(ax), "ay": float(ay), "az": float(az),  # m/s²
                    "gx": float(gx), "gy": float(gy), "gz": float(gz),  # rad/s
                    "mx": float(mx), "my": float(my), "mz": float(mz),  # µT
                    "tempC": float(tempC),
                    "ts": time.time(),
                }
            
            # Vérifier timeout
            if time.perf_counter() - t0 > timeout_s:
                self.timeouts += 1
                return None
            
            # Petite pause pour éviter de saturer le CPU
            time.sleep(0.001)
    
    def get_stats(self):
        """Retourne les statistiques de communication"""
        return {
            "total_packets": self.total,
            "bad_crc": self.bad_crc,
            "timeouts": self.timeouts,
            "error_rate": self.bad_crc / self.total if self.total > 0 else 0,
        }
    
    def __enter__(self):
        """Support du context manager"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Support du context manager"""
        self.close()


# ============================================================================
# TESTS
# ============================================================================

def test_hardware_uart():
    """Test simple de l'UART matériel"""
    print("="*70)
    print("TEST UART MATÉRIEL")
    print("="*70)
    print("\nConfiguration:")
    print("  - Arduino TX → Raspberry Pi GPIO 15 (RXD0, Pin 10)")
    print("  - Baudrate: 115200")
    print("  - Durée: 10 secondes\n")
    
    with ImuHardwareUart(port="/dev/ttyS0", baudrate=115200) as imu:
        print("Réception des données...\n")
        
        start_time = time.time()
        last_seq = None
        seq_errors = 0
        count = 0
        
        while time.time() - start_time < 10:
            m = imu.read_measurement(timeout_s=0.5)
            
            if m is None:
                print("⚠️  Timeout!")
                continue
            
            # Vérifier continuité de la séquence
            if last_seq is not None:
                expected = (last_seq + 1) % (2**32)
                if m["seq"] != expected:
                    seq_errors += 1
                    print(f"⚠️  Saut de séquence: {last_seq} → {m['seq']}")
            
            last_seq = m["seq"]
            count += 1
            
            # Affichage périodique
            if count % 100 == 0:
                print(f"✓ {count} paquets | "
                      f"seq={m['seq']} | "
                      f"acc=({m['ax']:.2f}, {m['ay']:.2f}, {m['az']:.2f}) m/s²",
                      end='\r')
        
        print("\n")
        
        # Statistiques finales
        stats = imu.get_stats()
        duration = time.time() - start_time
        
        print("="*70)
        print("RÉSULTATS:")
        print(f"  Durée: {duration:.1f} s")
        print(f"  Paquets reçus: {stats['total_packets']}")
        print(f"  Fréquence: {stats['total_packets']/duration:.1f} Hz")
        print(f"  Erreurs CRC: {stats['bad_crc']} ({stats['error_rate']*100:.2f}%)")
        print(f"  Timeouts: {stats['timeouts']}")
        print(f"  Sauts de séquence: {seq_errors}")
        
        if stats['bad_crc'] == 0 and stats['timeouts'] == 0 and seq_errors == 0:
            print("\n✅ COMMUNICATION PARFAITE!")
        elif stats['error_rate'] < 0.01:
            print("\n✓ Communication acceptable")
        else:
            print("\n⚠️  Problèmes de communication détectés")
            print("   → Vérifier le câblage")
            print("   → Vérifier que le baudrate correspond (Arduino et Pi)")


def test_with_stats():
    """Test avec affichage des statistiques en temps réel"""
    print("="*70)
    print("TEST AVEC STATISTIQUES TEMPS RÉEL")
    print("="*70)
    
    imu = ImuHardwareUart(port="/dev/ttyS0", baudrate=115200)
    imu.open()
    
    try:
        print("\nCtrl+C pour arrêter\n")
        last_display = time.time()
        
        while True:
            m = imu.read_measurement(timeout_s=0.5)
            
            if m is None:
                continue
            
            # Affichage toutes les secondes
            if time.time() - last_display >= 1.0:
                stats = imu.get_stats()
                print(f"Paquets: {stats['total_packets']:6d} | "
                      f"CRC OK: {stats['total_packets'] - stats['bad_crc']:6d} | "
                      f"Erreurs: {stats['bad_crc']:4d} | "
                      f"Timeouts: {stats['timeouts']:4d}")
                last_display = time.time()
    
    except KeyboardInterrupt:
        print("\n\nArrêt...")
    
    finally:
        stats = imu.get_stats()
        print(f"\nStatistiques finales:")
        print(f"  Total: {stats['total_packets']} paquets")
        print(f"  Taux d'erreur: {stats['error_rate']*100:.2f}%")
        imu.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "stats":
        test_with_stats()
    else:
        test_hardware_uart()