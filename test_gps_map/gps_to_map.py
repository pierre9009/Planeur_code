# gps_to_map.py
from gps_base import open_gps, wait_for_fix
import webbrowser
import time

def open_map(lat, lon):
    # URL OpenStreetMap centrée sur le point, zoom 18
    url = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=18/{lat}/{lon}"
    print(f"[MAP] Ouverture de la carte: {url}")
    webbrowser.open(url)


def main():
    # Ouvre et configure le GPS, cold start possible au lancement
    ser = open_gps(cold_start=True)

    try:
        # 1) On attend la première position valide
        lat, lon = wait_for_fix(ser, verbose=True)
        print(f"[GPS] Première position valide: {lat}, {lon}")

        # 2) On ouvre la carte sur ce point
        open_map(lat, lon)

        # 3) Ensuite, on continue à suivre les nouvelles positions
        print("[GPS] Suivi temps réel des positions (Ctrl+C pour quitter)")
        while True:
            # Ici on peut soit rappeler wait_for_fix, soit parser manuellement
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            if line.startswith("$GPRMC") or line.startswith("$GNRMC"):
                parts = line.split(",")
                if len(parts) < 7:
                    continue

                status = parts[2]
                lat_str = parts[3]
                lat_hemi = parts[4]
                lon_str = parts[5]
                lon_hemi = parts[6]

                if status != "A":
                    continue

                from gps_base import nmea_to_decimal_latlon
                lat, lon = nmea_to_decimal_latlon(lat_str, lat_hemi, lon_str, lon_hemi)
                if lat is not None and lon is not None:
                    print(f"[GPS] Position actuelle: {lat}, {lon}")
                    # Si tu veux, tu peux réouvrir la carte ou envoyer ces
                    # coordonnées à une interface graphique ou un serveur web.
                    # Pour ne pas ouvrir 1000 onglets, on se contente ici du print.

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nArrêt par l'utilisateur")
    finally:
        ser.close()
        # pas de GPIO.cleanup ici, il est dans gps_base si tu l’utilises comme script


if __name__ == "__main__":
    main()
