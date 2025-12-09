from micropyGPS import MicropyGPS
import serial

gps = MicropyGPS()
ser = serial.Serial('/dev/serial0', 9600, timeout=1)

while True:
    c = ser.read().decode('ascii', errors='ignore')
    if c:
        gps.update(c)
        if gps.fix_stat > 0:
            print(gps.latitude, gps.longitude)
