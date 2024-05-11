# Dataset
Download data from: https://www.kaggle.com/datasets/tommyngx/inbreast2012

# Structure  
|-- Data  
|&nbsp;--- Abnormal  
|&nbsp;--- Normal  
&nbsp;&nbsp;&nbsp;&nbsp;--Train  
&nbsp;&nbsp;&nbsp;&nbsp;--Test  

# Model Architecture
Encoder-Decoder

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
    python train_GAN.py
    ```

    Make sure to adjust any parameters or configurations in the `train_GAN.py` script according to your requirements before running it.

Ensure that you have a suitable Python environment set up and configured before proceeding with the steps above.

  
