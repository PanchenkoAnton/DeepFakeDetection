# DeepFakeDetection

Implementation of DeepFake-Detection-Challenge solution as a part of Master Thesis in Saint Petersburg State University.

## 1. Dependencies
To install all required Python packages you may use requirements.txt file with the terminal opened in the root repository directory:

`pip install -r requirements.txt`

## 2. Getting data
You may download desired raw data from [Kaggle competition data page](https://www.kaggle.com/c/deepfake-detection-challenge/data).

Be careful! The full training set is just over 470 GB. It is available as one giant file, as well as 50 smaller files, each ~10 GB in size. 

You must accept the competition's rules to gain access to any of the links. After that, according to the dataset authors, you may freely use it for creating deepfake detection solutions.


## 3. Dataset processing
To create convenient dataset (consists of aligned face images) from raw videos you can use `dataset_processing.py` script. That Python module extracts aligned face images from one .mp4 file. You should run it recursively with all .mp4 files as an input to process full dataset. For instance, you may use easy bash script for that.

## 4. Xception
Model definition is placed here: 

`models/xception.py`.

Code for the training may be found here:

`training/xception.py`.

Python module with some training settings is here:

`settings/xception.py`.

## 5. WSDAN
Model definition is placed here: 

`models/wsdan.py`.

Code for the training may be found here:

`training/wsdan.py`.

Python module with some training and model settings is here:

`settings/wsdan.py`.

