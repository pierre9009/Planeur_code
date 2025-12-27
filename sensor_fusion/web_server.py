import subprocess
import json
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO


app = Flask(__name__)
app.config['SECRET_KEY'] = 'imu_visualization_secret'
socketio = SocketIO(app, cors_allowed_origins="*")


def imu_reader_thread():
    """
    Thread qui lance le processus de fusion IMU en sous-processus
    et lit les données JSON depuis stdout.
    """
    try:
        # Lancer le processus de fusion IMU
        process = subprocess.Popen(
            ['python3', 'fus_manuel.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            universal_newlines=True
        )
        
        print("Processus IMU fusion démarré (PID: {})".format(process.pid))
        
        # Thread pour lire stderr (logs de débogage)
        def read_stderr():
            for line in process.stderr:
                print("[IMU FUSION] " + line.rstrip())
        
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
        
        # Lire stdout (données JSON)
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            
            try:
                # Parser le JSON
                orientation_data = json.loads(line)
                
                # Envoyer via WebSocket aux clients connectés
                socketio.emit('orientation_update', orientation_data)
                
            except json.JSONDecodeError as e:
                print(f"Erreur JSON: {e} - Line: {line}")
        
        # Si le processus se termine
        print("Processus IMU fusion terminé")
        process.wait()
        
    except Exception as e:
        print(f"Erreur dans imu_reader_thread: {e}")
        import traceback
        traceback.print_exc()


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/raw')
def raw_data():
    return render_template('raw.html')


@socketio.on('connect')
def handle_connect():
    print('Client WebSocket connecté')


@socketio.on('disconnect')
def handle_disconnect():
    print('Client WebSocket déconnecté')


if __name__ == '__main__':
    # Démarrage du thread qui lit les données IMU
    reader_thread = threading.Thread(target=imu_reader_thread, daemon=True)
    reader_thread.start()
    
    print("\n" + "="*70)
    print("Serveur Web démarré sur http://0.0.0.0:5000")
    print("Ouvrez cette adresse dans votre navigateur")
    print("="*70 + "\n")
    
    # Démarrage du serveur Flask
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)