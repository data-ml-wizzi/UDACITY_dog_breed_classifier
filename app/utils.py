import numpy as np
import random
from keras.preprocessing import image
from sklearn.datasets import load_files
from tensorflow.keras.utils import to_categorical
from glob import glob
from tqdm import tqdm


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