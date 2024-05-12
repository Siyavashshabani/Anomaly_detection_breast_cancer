# Dataset
Download data from: https://www.kaggle.com/datasets/tommyngx/inbreast2012

This dataset comprises 410 mammograms across 116 cases, including bilateral images from 95 cancer-diagnosed patients, thus capturing a wide array of breast disease manifestations, such as calcifications, masses, distortions, and asymmetries. The dataset provides images in both craniocaudal (CC) and mediolateral oblique (MLO) views and categorizes breast density according to the BI-RADS assessment into four levels, from entirely fat to extremely dense. The samples of normal and abnormal patches that has been used in train and test, respectively are given below.

![Dataset Images](https://github.com/sohaibcs1/Anomaly_detection_breast_cancer/blob/main/images/dataset.png)


# Model Architecture
# Autoencoder 
The autoencoder architecture consists of an encoder and decoder with four fully connceted layers. 

![Diagram](https://github.com/Siyavashshabani/Anomaly_detection_breast_cancer/blob/main/images/diagram.png)
# GAN
The proposed GAN network consists of a generator(Autoencoder ) and discriminator: 
![Diagram_gan](https://github.com/Siyavashshabani/Anomaly_detection_breast_cancer/blob/main/images/GAN_diagram.png)

# Structure  
|-- Data  
|&nbsp;--- Abnormal  
|&nbsp;--- Normal  
&nbsp;&nbsp;&nbsp;&nbsp;--Train  
&nbsp;&nbsp;&nbsp;&nbsp;--Test  

# Data preprocessing

You can find the data preprocessing code [here](https://github.com/Siyavashshabani/Anomaly_detection_breast_cancer/blob/main/preprocessing/preprocessing_Inbreast.ipynb).

## Running the Model

To run the model, follow these steps:

1. **Install Dependencies**: Ensure you have all the required dependencies installed. Navigate to the root directory of the project in your terminal and execute the following command:

    ```
    pip install -r requirements.txt
    ```

    This command will install all the necessary Python packages listed in the `requirements.txt` file.

2. **Navigate to the Training Directory**: Change your current directory to the `train` directory where the training scripts are located. Execute the following command:

    ```
    cd train
    ```

3. **Execute the Training Script**: Run the training script `train_GAN.py` to start the training process for the Generative Adversarial Network (GAN) model:

    ```
    python train_GAN.py --model AE_GAN
    ```

    Make sure to adjust any parameters or configurations in the `train_GAN.py` script according to your requirements before running it.

Ensure that you have a suitable Python environment set up and configured before proceeding with the steps above.

## Results
In this section, the reconstruction error for one abnormal patch is presented:
![result](https://github.com/Siyavashshabani/Anomaly_detection_breast_cancer/blob/main/images/result_img.png)



