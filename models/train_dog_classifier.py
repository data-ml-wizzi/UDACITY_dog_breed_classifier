#import basics
import os
import sys
import numpy as np
from tqdm import tqdm
import contextlib

#import data loading stuff
from sklearn.datasets import load_files
from glob import glob


#import tensorflow stuff
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint  
from keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as xc_ppi

#import evaluation
from sklearn.metrics import classification_report



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

def get_targets(data_path):
    """
    Loads the dog breed targets from the specified data directory.

    Args:
        data_path (str): Path to the dataset directory (e.g., 'dogImages/train').

    Returns:
        list: A list of dog breed targets for the data.
    """
    data = load_files(data_path)
    targets = to_categorical(np.array(data['target']), 133)
    
    return targets

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



def main():

    if len(sys.argv) == 1:
               
        # load train, test, and validation datasets
        print('-------------------------------------------------')
        print('Loading dog names and targets...')
        dog_names = get_dog_names('dogImages/train')
        train_targets = get_targets('dogImages/train')
        valid_targets = get_targets('dogImages/valid')
        test_targets = get_targets('dogImages/test')

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
        print('Saving the model...')
        model.save('saved_models/Xception_best.keras')

        print('-------------------------------------------------')
        print('Model build, trained, evaluated & saved')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()