# Quality-of-Life
This repository contains code that helps you to build a model which will predict the quality of life from satellite imagery.
You can find the more information regarding the model development here https://info.gramener.com/whitepaper-satellite-images-quality-of-life. The code is available under MIT License

# Execution
The zip folder contains all the necessary files to execute the project.

data\qol\sentinel.csv: has lat long information for USA counties

eo_download.py: download the satellite imagery

preprocess.py: process the images

srm_single_dataset.py: create train, validation and test data

srm_single_model.py: building the deep learning architechture 

srm_single_train.py: train the model

output.py: predicting the quality of life score


