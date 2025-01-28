#import basics
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import contextlib

#import data loading stuff
from sklearn.datasets import load_files
from glob import glob
import random  

# face & dog recognition stuff
import cv2
from keras.applications.resnet50 import ResNet50 as RN50
from keras.preprocessing import image    
from keras.applications.resnet50 import preprocess_input as rn_ppi


#import tensorflow stuff
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint  
from keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.applications.xception import Xception, preprocess_input
from keras.applications.xception import preprocess_input as xc_ppi

#import evaluation
from sklearn.metrics import classification_report








# OLD STUFF

#from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer


import pickle



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

def load_dataset(path):
    """
    Function to load datasets from a path and return a list of files and targets.
    
    Parameter:
        path:           text string of the path from the dataset to load
    
    Return:
        dog_files:      the files paths of the files in the dataset
        dog_targets:    the breeds of the respectives files
    """
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = to_categorical(np.array(data['target']), 133)
    
    return dog_files, dog_targets


def load_dog_data():
    """
    Function to load files and targets for train, validation and test sets,
    as well as all the dog names. At the end print statistics.
    
    Parameter:
        None
    
    Return:
        train_files: the files paths to the train data
        train_targets: the breed of the train file
        valid_files: the file paths to the validation data
        valid_targets: the breed of the validation files
        test_files: the files paths to the testing data
        test_targets: the breed of the testing files
    """
    
    # load train, test, and validation datasets
    train_files, train_targets = load_dataset('dogImages/train')
    valid_files, valid_targets = load_dataset('dogImages/valid')
    test_files, test_targets = load_dataset('dogImages/test')

    # load list of dog names
    dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

    # print statistics about the dataset
    #print('-------- DOG DATA STATISTICS ----------')
    print('>> There are %d total dog categories.' % len(dog_names))
    print('>> There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
    print('>> There are %d training dog images.' % len(train_files))
    print('>> There are %d validation dog images.' % len(valid_files))
    print('>> There are %d test dog images.'% len(test_files))
    

    return train_files, train_targets,\
           valid_files, valid_targets,\
           test_files, test_targets,\
           dog_names


def load_human_data():
    """
    Function to load the human files
    
    Parameter:
        None
    
    Return:
        human_files: file paths to the pictures of humans
    """

    random.seed(8675309)

    # load filenames in shuffled human dataset
    human_files = np.array(glob("lfw/*/*"))
    random.shuffle(human_files)

    # print statistics about the dataset
    print('>> There are %d total human images.' % len(human_files))

    return human_files

def face_detector(img_path):
    """
    Function to determine of the image of a retrieved file path contains a human face
    
    Parameter:
        img_path: text string of the file path to the image
    
    Return:
        True:  if a face is detected
        False: if no face is detected
    """
    #retrieve the pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    # load color (BGR) image
    img = cv2.imread(img_path)
    # convert BGR image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find faces in image
    faces = face_cascade.detectMultiScale(gray)

    return len(faces) > 0               

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

def paths_to_tensor(img_paths):
    """
    Function to load multiple images, turn them into a 4D Tensor and return a list of 4D Tensors.
    
    Parameter:
        img_paths: text strings of the file paths to the images
    
    Return:
        a numpy array of 4D Tensors from all images from img_paths
    """
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    
    return np.vstack(list_of_tensors)



def dog_detector(img_path):
    """
    Function to to detect dogs in a picture
    
    Parameter:
        img_path: text string to the relevant picture to be checked
    
    Return:
        prediction: True (a dog) / False (no dog)
    """
    # define ResNet50 model
    ResNet50_model_dog_class = RN50(weights='imagenet')
    
    # load image, transform to a tensor and preprocess it for ResNet50
    img = rn_ppi(path_to_tensor(img_path))
    
    # make a prediction
    with suppress_stdout_stderr():
        prediction = np.argmax(ResNet50_model_dog_class.predict(img))
    
    return ((prediction <= 268) & (prediction >= 151)) 


def extract_Xception(tensor):
	"""
    Function to to detect dogs in a picture
    
    Parameter:
        img_path: text string to the relevant picture to be checked
    
    Return:
        prediction: True (a dog) / False (no dog)
    """
	return Xception(weights='imagenet', include_top=False).predict(xc_ppi(tensor))


def build_model(shape):
    """
    Function to build the Machine Learning Model for prediction. Transfer learning is used.
    As a base model we use the Xception model.
    
    Parameter:
        shape: int of the input shape from the bottleneck features
    
    Return:
        model: a model based on the Xception modified for the 133 breed dog classification problem
    """    
    # create the model
    Xception_model = Sequential()
    Xception_model.add(GlobalAveragePooling2D(input_shape=shape))
    Xception_model.add(Dense(133, activation='softmax'))

    # give a summary
    Xception_model.summary()

    # compile the model
    Xception_model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])
    
    
    return Xception_model

def evaluate_model(model, X_test, Y_test, names):
    """
    Function to evaluate the Machine Learning Model.
    
    Parameter:
        model: Builded Model
        X_test: Input test data
        Y_test: Output test data
    
    Return:
        NONE
    """ 
    # Load the model weights with the best validation loss.
    model.load_weights('saved_models/weights.best.Xception.h5')
    
    # Predict values with model:
    y_pred = list()

    for feature in tqdm(X_test):
        with suppress_stdout_stderr():
            y_pred.append(np.argmax(model.predict(np.expand_dims(feature, axis=0))))

    # classification report on test data
    print(classification_report(np.array(y_pred),
                                np.argmax(Y_test, axis=1),
                                target_names=names))
    
    return

def save_model(model, model_filepath):
    """
    Save the model into a PICKLE-File
    
    Parameter:
        model: Model to save
        model_filepath: Path 
    
    Return:
        NONE
    """     
    #pickle.dump(model, open(model_filepath, 'wb') )
   
    return


def main():
    # Set log level
    

    if len(sys.argv) == 1:
               
        # load train, test, and validation datasets
        print('-------------------------------------------------')
        print('Loading dog data into train, test & validation sets...')
        train_files, train_targets,\
        valid_files, valid_targets,\
        test_files, test_targets,\
        dog_names = load_dog_data()

        print('-------------------------------------------------')
        print('Loading file paths to the pictures of humans...')
        human_files = load_human_data()

        print('-------------------------------------------------')
        print('Creating test subsets...')
        human_files_short = human_files[:100]
        dog_files_short = train_files[:100]

        print('Testing "Dog Detecter" and "Face Detector"...')
        for dog in dog_files_short:
            dog_detector(dog)
            print('"Dog Detecter" works...')
            break

        for human in human_files_short:
            face_detector(human)
            print('"Face Detecter" works...')
            break

        print('-------------------------------------------------')
        print('Loading bottleneck features for Xception Model...')
        bottleneck_features = np.load('bottleneck_features/DogXceptionData.npz')
        train_Xception = bottleneck_features['train']
        valid_Xception = bottleneck_features['valid']
        test_Xception = bottleneck_features['test']
        
        print('-------------------------------------------------')
        print('Building model...')
        model = build_model(train_Xception.shape[1:])
        
        print('-------------------------------------------------')
        print('Training model...')
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.Xception.h5', 
                               verbose=1, save_best_only=True)

        model.fit(train_Xception, train_targets, 
                  validation_data=(valid_Xception, valid_targets),
                  epochs=20, batch_size=20, callbacks=[checkpointer], verbose=10)
        
        print('-------------------------------------------------')
        print('Evaluating model...')
        evaluate_model(model, test_Xception, test_targets, dog_names)

        print('-------------------------------------------------')
        print('Model build, trained & evaluated')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()