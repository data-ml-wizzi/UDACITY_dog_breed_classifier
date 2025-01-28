# BASICS
import os
import sys
import numpy as np
from werkzeug.utils import secure_filename
import random
import contextlib  

# LOADING DATA
from glob import glob
from sklearn.datasets import load_files
from tensorflow.keras.utils import to_categorical

# FACE & DOG DETECTOR
import cv2

# FLASK
from flask import Flask, request, render_template, redirect, url_for

# TENSORFLOW & MODEL
from tensorflow.keras.models import load_model
from keras.applications.resnet50 import preprocess_input as rn_ppi
from keras.preprocessing import image  
from keras.applications.resnet50 import ResNet50 as RN50
from keras.applications.xception import Xception, preprocess_input

# PICTURE
from PIL import Image





# Init of Flask App
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained breed classification model
model = load_model('saved_models/Xception_best.keras')

# retrieve the pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

# pretrained ResNet50 model for dog detector
ResNet50_model = RN50(weights='imagenet')




# Define a context manager to suppress stdout and stderr for a cleaner output
@contextlib.contextmanager
def suppress_stdout_stderr():
    """
    A context manager to suppress stdout and stderr for a cleaner output
    
    Parameter: None
    
    Return: None
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def get_dog_names(data_path):
    """
    Loads the dog breed names from the specified training data directory.

    Args:
        data_path (str): Path to the training dataset directory (e.g., 'dogImages/train').

    Returns:
        list: A list of dog breed names extracted from the directory structure.
    """
    dog_names = [item[20:-1] for item in sorted(glob(f"{data_path}/*/"))]
    
    return dog_names


def path_to_tensor(img_path):
    """
    Function to load the image from img_path and turn it into a 4D tensor.
    
    Parameter:
        img_path: text string of the file path to the image
    
    Return:
        4D Tensor of the image from img_path
    """
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


def extract_Xception(tensor):
	"""
    Function to extract the Xcpetion bottleneck features
    
    Parameter:
        tensor: a tensor
    
    Return:
        the bottleneck features
    """
	return Xception(weights='imagenet', include_top=False).predict(preprocess_input(tensor))


def face_detector(img_path, face_cascade):
    """
    Function to determine if the image of a retrieved file path contains a human face
    
    Parameter:
        img_path: text string of the file path to the image
        face_cascade: the pretrained face detector
    
    Return:
        True:  if a face is detected
        False: if no face is detected
    """
    # load color (BGR) image
    img = cv2.imread(img_path)
    # convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find faces in image
    faces = face_cascade.detectMultiScale(gray)

    return len(faces) > 0               


def dog_detector(img_path, model):
    """
    Function to to detect dogs in a picture
    
    Parameter:
        img_path: text string to the relevant picture to be checked
    
    Return:
        prediction: True (a dog) / False (no dog)
    """
    # load image, transform to a tensor and preprocess it for ResNet50
    img = rn_ppi(path_to_tensor(img_path))
    # make a prediction
    with suppress_stdout_stderr():
        prediction = np.argmax(model.predict(img))
    
    return ((prediction <= 268) & (prediction >= 151)) 


def classify_dog_breed(img_path, model):
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model.predict(bottleneck_feature)
    #return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]


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
            
            # save the uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # predict the breed
            breed = classify_dog_breed(file_path, model)

            # go trough the requested logic
            if face_detector(file_path, face_cascade):
                picture_type = 'human'
            elif dog_detector(file_path, ResNet50_model):
                picture_type = 'dog'
            else:
                picture_type = 'neither'
                print("ERROR: neither a human nor a dog were recognized")                            

            # return the attained reults to the index.html
            return render_template('index.html', 
                                   breed=breed, 
                                   type=picture_type, 
                                   img_url=file_path)
        
    return render_template('index.html', breed=None, type=None, img_url=None)



if __name__ == '__main__':
    dog_names = get_dog_names('dogImages/train')
    app.run(debug=True)
