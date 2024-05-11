# Dataset
Download data from: https://www.kaggle.com/datasets/tommyngx/inbreast2012

This dataset comprises 410 mammograms across 115 cases, including bilateral images from 95 cancer-diagnosed patients, thus capturing a wide array of breast disease manifestations, such as calcifications, masses, distortions, and asymmetries. The dataset provides images in both craniocaudal (CC) and mediolateral oblique (MLO) views and categorizes breast density according to the BI-RADS assessment into four levels, from entirely fat to extremely dense. 

# Structure  
|-- Data  
|&nbsp;--- Abnormal  
|&nbsp;--- Normal  
&nbsp;&nbsp;&nbsp;&nbsp;--Train  
&nbsp;&nbsp;&nbsp;&nbsp;--Test  


![Method Image]([https://github.com/sohaibcs1/Anomaly_detection_breast_cancer/raw/main/method.png](https://github.com/sohaibcs1/Anomaly_detection_breast_cancer/blob/main/train.png)


# Model Architecture
The structure of the proposed method consists of two main sections. The left section includes encoder and decoder blocks for transforming data into a low-dimensional space, while the central section is dedicated to the core diffusion model.
<br>
![Method Image](https://github.com/sohaibcs1/Anomaly_detection_breast_cancer/raw/main/method.png)


# How to Run Model
* pip install requirements.txt
* cd train
* python train_GAN.py
  


