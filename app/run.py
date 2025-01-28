from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from models import classify_dog_breed

# Initialisiere Flask-App
app = Flask(__name__)

# Konfiguration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Prüfe, ob die Post-Anfrage eine Datei enthält
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # Rufe den Klassifizierer auf und erhalte das Ergebnis
            breed = classify_dog_breed(file_path)
            return render_template('index.html', breed=breed)
    return render_template('index.html', breed=None)

def classify_dog_breed(image_path):
    # Füge hier den Code für die Hundebrüten-Klassifikation ein
    # und gebe die erkannte Hundebrüte als String zurück
    return "Unbekannte Rasse"

if __name__ == '__main__':
    app.run(debug=True)
