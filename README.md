# AI-Project
This project is a prototype self-checkout system that can scan items and automatically check a customer’s age when they try to buy a restricted product. When such an item is scanned, the system turns on the camera, captures the customer’s face, and estimates their age.
If the person looks 25 or older, the purchase continues normally.
If they look younger than 25, the system asks for an admin to verify the ID.
The interface includes item scanning (via QR codes), a live camera window for age checks, and an admin approval screen. It shows how age verification could be made quicker and more convenient in self-checkout machines.
# Backend
Behind the scenes, the system uses a model trained on face-image datasets to estimate a person’s age. We tested two versions: a basic CNN and a deeper ResNet-50 model. Both take a face image, analyze its features, and output a predicted age, but the ResNet model performed better and was used in the final prototype.

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
![](images/eda (1).png)
![](images/eda (2).png)
![](images/eda (3).png)
![](images/eda (4).png)
![](images/eda (5).png)
![](images/eda (6).png)
![](images/training (1).png)
![](images/training (1).png)


