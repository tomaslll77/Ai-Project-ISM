# AI-Project
This project is a prototype self-checkout system that can scan items and automatically check a customer’s age when they try to buy a restricted product. When such an item is scanned, the system turns on the camera, captures the customer’s face, and estimates their age.
If the person looks 25 or older, the purchase continues normally.
If they look younger than 25, the system asks for an admin to verify the ID.
The interface includes item scanning (via QR codes), a live camera window for age checks, and an admin approval screen. It shows how age verification could be made quicker and more convenient in self-checkout machines.
# The main things
Behind the scenes, the system uses a model trained on face-image datasets to estimate a person’s age. We tested two versions: a basic CNN and a deeper ResNet-50 model. Both take a face image, analyze its features, and output a predicted age, but the ResNet model performed better and was used in the final prototype.
# How to use?
The required libraries are provided in requirements.txt
Download our best model checkpoint final_resnet50_model.ckpt from releases(v1.0.0) and run gui_savitarna.py.
The training part of the model, architecture itself is provided in a notebook *resnet_training_notebook.ipynb*

The training data was created by combining the UTKFace and APPA-REAL datasets, which include thousands of labeled face photos. 
# DATASET LINK FOR DOWNLOAD
https://www.kaggle.com/datasets/moritzm00/utkface-cropped?resource=download
UTKFace (Aligned & Cropped)
https://www.kaggle.com/datasets/abhikjha/appa-real-face-cropped
APPA-REAL Cropped face dataset
# Dataset labels (UTKFace)
The labels of each face image is embedded in the file name, formated like [age]_[gender]_[race]_[date&time].jpg
[age] is an integer from 0 to 116, indicating the age
[gender] is either 0 (male) or 1 (female)
[race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others.
[date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace.
# Dataset labels (APPA-REAL)
The label of each face image is only age, which can be found by downloading the dataset and opening labels.csv
# Results from training and basic EDA
![eda1](https://github.com/tomaslll77/Ai-Project-ISM/blob/e0f692f149c2f7a8ad8903b04bea47d7dd1a7695/eda_training_images/eda%20(1).png)
![eda2](https://github.com/tomaslll77/Ai-Project-ISM/blob/e0f692f149c2f7a8ad8903b04bea47d7dd1a7695/eda_training_images/eda%20(2).png)
![eda3](https://github.com/tomaslll77/Ai-Project-ISM/blob/e0f692f149c2f7a8ad8903b04bea47d7dd1a7695/eda_training_images/eda%20(3).png)
![eda4](https://github.com/tomaslll77/Ai-Project-ISM/blob/e0f692f149c2f7a8ad8903b04bea47d7dd1a7695/eda_training_images/eda%20(4).png)
![eda5](https://github.com/tomaslll77/Ai-Project-ISM/blob/e0f692f149c2f7a8ad8903b04bea47d7dd1a7695/eda_training_images/eda%20(5).png)
![eda6](https://github.com/tomaslll77/Ai-Project-ISM/blob/e0f692f149c2f7a8ad8903b04bea47d7dd1a7695/eda_training_images/eda%20(6).png)
![train1](https://github.com/tomaslll77/Ai-Project-ISM/blob/e0f692f149c2f7a8ad8903b04bea47d7dd1a7695/eda_training_images/training%20(1).png)
![train2](https://github.com/tomaslll77/Ai-Project-ISM/blob/e0f692f149c2f7a8ad8903b04bea47d7dd1a7695/eda_training_images/training%20(2).png)


