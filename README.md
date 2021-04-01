# Dis-: not a problem!

Dis-: not a problem! - Progetto di tesi magistrale (Informatica Umanistica) di CHIARA PUGLIESE

Requirements:

1. TensorFlow v. 2.3.0
2. Keras v. 2.4.3
3. Flask v. 1.1.2
4. nltk v. 3.2.5

The folder is organized as follow:

Dis-: not a problem:
|
|---Detection&Segmentation of images
|---Handwriting recognition
|---Red flags of dysorthography
|---Website

HANDWRITING DETECTION AND IMAGE SEGMENTATION folder.

Files in this folder are .ipynd, so they are jupyter notebook files.
In this folder there are scripts for handwritining detection, starting from images and for image segmentation in words and characters.

HANDWRITING RECOGNITION folder

Files in this folder are .ipynd, so they are jupyter notebook files.
In this folder there are scripts for handwriting recognition.
In the folder Models there are .h5 files of tested models.

The folder is organized as follow:

HANDWRITING RECOGNITION
|
|---Models
|	|
|	|---Images of models
|---Uppercase Handwriting Recognition
|	|
|	|---Data Augmentation
|	|---No Data Augmentation
|---Lowercase Handwriting Recognition
	|
	|---Data Augmentation
	|---No Data Augmentation
	
RED FLAGS DETECTION folder

Files in this folder are .ipynd, so they are jupyter notebook files.
In this folder there are scripts for red flags detection.
In the folder Models there are .pickle and .h5 files of tested models.

RED FLAGS DETECTION
|
|---Models
|
|---Dataset
|
|---Red flags detection
	
WEBSITE folder

In this folder there are the web application create with Flask. To open the website, run "sito.py"
The folder is organized as follow:

WEBSITE
|
|---static
|	|
|	|---fonts
|	|---images
|	|---js
|
|---templates
