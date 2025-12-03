# simulation.py

import math
import csv

# paramètres simulation
dt = 0.2  # s
total_time = 150  # secondes simulées
speed = 15  # m/s (vitesse sol avion)
turn_rate_deg = 6  # deg/s en holding
HOLD_RADIUS = 50  # m

# waypoint (target) = origine
target_x, target_y = 0, 0

# position initiale (1000 m au nord)
x, y = 1000, 0

# heading initial
heading_deg = 180  # vers le sud

with open("sim_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["time", "x", "y", "heading_deg", "distance", "mode"])

    time_s = 0
    while time_s <= total_time:
        # distance au waypoint
        distance = math.hypot(target_x - x, target_y - y)

        if distance > HOLD_RADIUS:
            mode = "DIRECT"
            # cap vers waypoint
            target_heading_deg = math.degrees(math.atan2(-y, -x)) % 360
            # correction instantanée du heading vers cible
            heading_deg = target_heading_deg
        else:
            mode = "HOLDING"
            # rotation constante (virage à gauche)
            heading_deg = (heading_deg + turn_rate_deg * dt) % 360

        # déplacement selon heading
        rad = math.radians(heading_deg)
        x += speed * math.sin(rad) * dt
        y += speed * math.cos(rad) * dt

        writer.writerow([time_s, x, y, heading_deg, distance, mode])
        time_s += dt

print("Simulation terminée. Données enregistrées dans sim_data.csv")
