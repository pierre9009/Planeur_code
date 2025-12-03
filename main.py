# main.py

import time
import math
from sensors import read_gps, read_pressure_alt, read_imu_heading_deg
from guidance import guidance_direct_to

def main_loop():
    home_lat_rad = None
    home_lon_rad = None
    target_lat_rad = None
    target_lon_rad = None
    target_alt = None

    print("Démarrage module path planning...\n")

    while True:
        gps_lat_deg, gps_lon_deg, gps_alt = read_gps()
        baro_alt = read_pressure_alt()
        heading_deg = read_imu_heading_deg()

        lat_rad = math.radians(gps_lat_deg)
        lon_rad = math.radians(gps_lon_deg)
        heading_rad = math.radians(heading_deg)

        # initialisation home/target
        if home_lat_rad is None:
            home_lat_rad = lat_rad
            home_lon_rad = lon_rad
            target_lat_rad = home_lat_rad
            target_lon_rad = home_lon_rad
            target_alt = baro_alt
            print(f"Point HOME défini: {gps_lat_deg:.6f}, {gps_lon_deg:.6f}, alt {target_alt:.1f} m")

        heading_target, heading_error, dist, mode = guidance_direct_to(
            lat_rad, lon_rad, heading_rad,
            home_lat_rad, home_lon_rad,
            target_lat_rad, target_lon_rad
        )

        print(
            f"MODE {mode:<8} | DIST {dist:6.1f} m | "
            f"HDG {heading_deg:6.1f}° → {math.degrees(heading_target):6.1f}° "
            f"(err={math.degrees(heading_error):6.1f}°)"
        )

        time.sleep(0.2)  # 5 Hz par exemple

if __name__ == "__main__":
    main_loop()
