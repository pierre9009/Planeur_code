# state.py

import math

R_EARTH = 6371000.0  # m

def gps_to_local(lat_rad, lon_rad, home_lat_rad, home_lon_rad):
    dlat = lat_rad - home_lat_rad
    dlon = lon_rad - home_lon_rad
    x_north = dlat * R_EARTH
    y_east = dlon * R_EARTH * math.cos(home_lat_rad)
    return x_north, y_east


def wrap_angle(angle_rad):
    """Ramène angle dans [-pi, pi]."""
    while angle_rad > math.pi:
        angle_rad -= 2 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2 * math.pi
    return angle_rad
