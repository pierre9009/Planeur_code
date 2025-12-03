# visualisation.py

import pandas as pd
import matplotlib.pyplot as plt

# lecture de la simulation
df = pd.read_csv("sim_data.csv")

# configuration graphique
plt.figure(figsize=(10, 10))
plt.plot(df["x"], df["y"], label="Trajectoire")

# affichage waypoint
plt.scatter(0, 0, color="red", label="Waypoint")

# zones de holding
circle = plt.Circle((0, 0), 50, edgecolor="gray", facecolor="none", linestyle="--")
plt.gca().add_artist(circle)

# mode
for i, mode in enumerate(df["mode"].unique()):
    subset = df[df["mode"] == mode]
    plt.plot([], [], label=f"MODE {mode}")

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Trajectoire avion et zones de guidage")
plt.legend()
plt.grid(True)
plt.axis("equal")

plt.show()


# analyse du heading
plt.figure(figsize=(12, 5))
plt.plot(df["time"], df["heading_deg"], label="Heading avion")
plt.xlabel("Temps (s)")
plt.ylabel("Heading (°)")
plt.title("Évolution du cap en fonction du guidage")
plt.grid(True)
plt.legend()

plt.show()
