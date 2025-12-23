#!/bin/bash

SERVICE_NAME="pigpiod"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

echo "Création du service systemd pour pigpiod..."

sudo tee "$SERVICE_PATH" > /dev/null <<EOF
[Unit]
Description=Pigpio Daemon
After=network.target

[Service]
Type=forking
ExecStart=/usr/local/bin/pigpiod
ExecStop=/usr/local/bin/pigpiod -k
Restart=always

[Install]
WantedBy=multi-user.target
EOF

echo "Rechargement de systemd..."
sudo systemctl daemon-reexec
sudo systemctl daemon-reload

echo "Activation du service..."
sudo systemctl enable ${SERVICE_NAME}

echo "Démarrage du service..."
sudo systemctl start ${SERVICE_NAME}

echo "Statut du service :"
sudo systemctl status ${SERVICE_NAME} --no-pager
