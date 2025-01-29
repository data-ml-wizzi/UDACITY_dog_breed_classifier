# Project "CAPSTONE >> Dog Breed Classifier"
This project ist part of my "Data Science" Nanodegree on Udacity and my final submission.

# Table of Contents

1. [Project Overview & Motivation](#motivation)
2. [Installations](#installations)
2. [Data](#data)
3. [File Descriptions](#files)
4. [Instructions](#instructions)
5. [Results](#results)


# Project Overview and Motivation <a name="motivation"></a>
As part of the "Data Science" Nanodegree on Udacity we shall show our skills with creating an Convolutional Neural Network (CNN) Pipeline to classify dog breed and identify humand-dog resemblance. For the user a front end shall be supplied to upload the picture and then retrieve the breed of the dog (or the resemblance of the human). If neither a human or a dog is uploaded this shall be recognized and stated.

To achieve this I was guided towards the topic by working trough a Jupyter Notebook. There I did...

1. load the data and some basic data exploration and visualizations as well as data preparations (resizing and normalising) for the CNN models
2. build a human face detector using OpenCV's pre-trained Haar cascades
3. build a dog detector based on a pretrained ResNet50 Network
4. build an own CNN
5. build an CNN with transfer learning using the VGG16 CNN with its respective bootleneck features
6. build an CNN with transfer learning using the Xception CNN with its respective bootleneck features

Then I setup a train_dog_classifier with the learning from 6. which uses Transfer Learning from the Xception CNN cause it performed the best according to the metrics accuracy, precision and F1-score. Here the last Layer is defined by myself, this last layers then fitted to the training data and the retrieved model stored.

Furthermore a Flask App was build as an Front End for the Users. Here they can Upload a picture and then according to the requested logic it is determined it the picture contains a human, dog or neither. The uploaded image is displayed and in case of an detected dog or human its breed is classified.

# Installations <a name="installations"></a>

Python 3.12 was used. Herefore the following Packages have to be installed.

<ul>
    <li>Flask           == 3.1.0
    <li>h5py            == 3.12.1
    <li>ipykernel       == 6.29.5
    <li>ipython         == 8.31.0
    <li>keras           == 3.8.0
    <li>matplotlib      == 3.10.0
    <li>numpy           == 2.0.2
    <li>opencv-python   == 4.11.0.86
    <li>pandas          == 2.2.3
    <li>pillow          == 11.1.0
    <li>requests        == 2.32.3
    <li>scikit-lear     == 1.6.1
    <li>scipy           == 1.15.1
    <li>seaborn         == 0.13.2
    <li>tensorflow      == 2.18.0
    <li>tqdm            == 4.67.1
</ul>

# Data <a name="data"></a>
The Data for the dogs contains a training, validation and testing set. In total 8351 images. </br>

<ul>
  <li>training:     6680 images
  <li>validation:   835 images
  <li>testing:      836 images  
</ul>

The datasets contain a total of 133 breeds of dogs. The images are not spread evenly over these breeds. Therefore probably some breeds can be classified with higher accuracy, precision and f1-score than others.

<br>
    <div align="center">
	    <img src="https://github.com/data-ml-wizzi/UDACITY_dog_breed_classifier/blob/main/images/breed_distribution_datasets.png">
    </div>
    <div align="center">
	    <i>Figure 3 – Distribution of the images in the dataset over the breeds</i>
    </div>
<br>

Furthermore the image sizes vary quite a bit but at least the variation is evenly distributed over the different data sets, altough not obvious on first sight.


<br>
    <div align="center">
	    <img src="https://github.com/data-ml-wizzi/UDACITY_dog_breed_classifier/blob/main/images/image_dim_stats.png">
    </div>
    <div align="center">
	    <i>Figure 3 – Distribution of the images sizes in the datasets</i>
    </div>
<br>


# File Descriptions <a name="files"></a>

```
app
 |--- templates
 |       |---index.html                 <- Main page of the web app
 |--- run.py                            <- flask file that runs the app
 |--- utils.py                          <- helper functions

bottleneck_features
 |--- None                              <- download file to from instructions part 4

dog_images
 |--- None                              <- download files to from instructions part 2

haarcascades
 |--- haarcascade_frontalface_alt.xml   <- pretrained face detection 

lfw
 |--- None                              <- download files to from instructions part 3

models
 |--- train_dog_classifier.py           <- file to transfer learn a pretrained Xception CNN

saved_models
 |--- weights.best.from_scratch.h5      <- saved weights from notebook part 3
 |--- weights.best.VGG16.h5             <- saved weights from notebook part 4
 |--- weights.best.Xception.h5          <- saved weights from notebook part 5 
 |                                         OR train_dog_classifier.py
 |--- Xception_best.keras               <- best trained Xception model from train_dog_classifier.py
 ```

# Instructions <a name="instructions"></a>

1. Clone or download the repo

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Donwload the [Xception bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. Run the following commands in the project's root directory to set up your model.

    - To run ML pipeline that trains the classifier and saves its best version </br>
        
        `python models/train_dog_classifier.py`

6. Change to the app directory: `cd app`

7. Run your web app: `python run.py`

7. Click the `PREVIEW` button in the Terminal to open the homepage via http://loclhost:3000 or http://127.0.0.1:3000

8. Upload a picture and let the breed be classified. E voila.


# Results <a name="results"></a>
In the web app (Figure 1) the user can upload an image. 

<br>
    <div align="center">
	    <img src="https://github.com/data-ml-wizzi/UDACITY_dog_breed_classifier/blob/main/images/flask_app_landing.png">
    </div>
    <div align="center">
	    <i>Figure 1 – Screenshot of the Web-App Landing Page</i>
    </div>
<br>

Once the user hits "Get your Breed" the image is uploaded, the breed is classified and a human-face and dog detection runs on the image. The Image is displayed at all times. If a dog or human are recognized the (resembling) breed the dog/human are welcomed and they are told their breed.  

<br>
    <div align="center">
	    <img src="https://github.com/data-ml-wizzi/UDACITY_dog_breed_classifier/blob/main/images/flask_app_result.png">
    </div>
    <div align="center">
	    <i>Figure 2 – Screenshot of the Web-App after an image has been uploaded</i>
    </div>
<br>