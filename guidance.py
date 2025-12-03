# guidance.py

import math
from state import gps_to_local, wrap_angle

# paramètres
ROLL_MAX_DEG = 25
ROLL_MAX = math.radians(ROLL_MAX_DEG)
HOLD_RADIUS = 50.0  # m, rayon du cercle quand on est au point

def guidance_direct_to(lat_rad, lon_rad, heading_rad,
                       home_lat_rad, home_lon_rad,
                       target_lat_rad, target_lon_rad):
    """
    Retourne consigne heading_target (pas roll !)
    et mode ('DIRECT' ou 'HOLDING')
    """
    # position locale
    x, y = gps_to_local(lat_rad, lon_rad, home_lat_rad, home_lon_rad)
    tx, ty = gps_to_local(target_lat_rad, target_lon_rad, home_lat_rad, home_lon_rad)

    dx = tx - x
    dy = ty - y
    distance = math.hypot(dx, dy)

    if distance > HOLD_RADIUS:
        # GUIDAGE DIRECT
        bearing_target = math.atan2(dy, dx)
        mode = "DIRECT"
    else:
        # ON EST AU DESSUS -> ON TOURNE AUTOUR DU POINT
        # stratégie: ajouter un angle de rotation en continu
        # ici on choisit un virage à gauche en faisant +90° sur la direction vers le point
        bearing_target = math.atan2(dy, dx) + math.radians(90)
        mode = "HOLDING"

    bearing_target = wrap_angle(bearing_target)
    heading_error = wrap_angle(bearing_target - heading_rad)

    # ici on ne sort PAS roll, juste la consigne de cap
    # la correction se fera en low-level
    return bearing_target, heading_error, distance, mode
