import argparse, os, sys
import numpy as np
import imageio
from scipy import ndimage

import torch
from torchvision.utils import save_image
import torch.nn as nn

from models.VAE import VAE
from models.AE import AE
from models.UNet import UNet
from models.AE_384 import AE_384
from models.AE_GAN import AE_GAN

from models.AE_head import AE_head
import matplotlib.pyplot as plt 
from utils.utils import get_interpolations
from utils.utils import loss_plot, plots
import random

import argparse, os, sys
import numpy as np
import imageio
from scipy import ndimage

import torch
from torchvision.utils import save_image
import torch.nn as nn

from models.VAE import VAE
from models.AE import AE
from models.UNet import UNet
from models.AE_384 import AE_384
from models.AE_GAN import AE_GAN

from models.AE_head import AE_head
import matplotlib.pyplot as plt 
from utils.utils import get_interpolations
from utils.utils import loss_plot, plots
import random
from dataloaders.dataloaders import create_data_loaders
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# Define a function to pass data through the generator and discriminator
def pass_data_through_models(data_loader, model, device, data_type= "dataloader"):
    all_outputs = []
    batch_size = 64
    # device = args.cuda
    if data_type=="dataloader":
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                # print(data.shape)
                data = data.float().to(device)
                outputs = model(data)
                all_outputs.append(outputs)
    else:
        with torch.no_grad():
            for i in range(0, len(data_loader), batch_size):
                batch_data = data_loader[i:i+batch_size].float().to(device)  # Move batch data to device
                outputs = model(batch_data)
                all_outputs.append(outputs)
    return torch.cat(all_outputs, dim=0)

# Calculate accuracy
def calculate_accuracy(outputs, labels):
    # Convert probabilities to binary predictions
    predictions = (outputs > 0.5).float()
    # Compare predictions with ground truth labels
    correct_predictions = (predictions == labels).float()
    # Calculate accuracy
    accuracy = correct_predictions.mean().item() * 100
    return accuracy

def get_output(generator,discriminator, loader):
# Pass data through the generator to generate fake data
    fake_data = pass_data_through_models(loader, generator, device, data_type="dataloader")

    print("shape of fake data------------:", fake_data.shape)
    # Pass real and fake data through the discriminator for classification
    real_outputs = pass_data_through_models(loader, discriminator, device, data_type="dataloader")
    fake_outputs = pass_data_through_models(fake_data, discriminator, device, data_type="data")

    print("shape of fake_outputs------------:", fake_outputs.shape)
    print("shape of real_outputs------------:", real_outputs.shape)

    # Create labels for real and fake data (1 for real, 0 for fake)
    real_labels = torch.ones_like(real_outputs)
    fake_labels = torch.zeros_like(fake_outputs)

    # Concatenate real and fake data and labels
    all_outputs = torch.cat([real_outputs, fake_outputs], dim=0)
    all_labels = torch.cat([real_labels, fake_labels], dim=0)


    # Assuming real_labels and fake_labels are ground truth labels for real and fake data
    # real_outputs and fake_outputs are discriminator outputs for real and fake data respectively
    accuracy_real = calculate_accuracy(real_outputs, real_labels)
    accuracy_fake = calculate_accuracy(fake_outputs, fake_labels)

    print(f'Accuracy for real data: {accuracy_real:.2f}%')
    print(f'Accuracy for fake data: {accuracy_fake:.2f}%')

    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(all_labels.cpu(), all_outputs.cpu())
    roc_auc = auc(fpr, tpr) 
        
    return fpr, tpr, roc_auc



def acu_roc(autoenc,ae_384, loader):

    ### GAN output 
    fpr_gan, tpr_gan, roc_auc_gan = get_output(autoenc.generator, autoenc.discriminator, loader)

    ### GAN output 
    fpr_ae, tpr_ae, roc_auc_ae = get_output(ae_384.model, autoenc.discriminator, loader)


    # Plot ROC curve for GAN output
    plt.plot(fpr_gan, tpr_gan, color='darkorange', lw=2, label=f'GAN ROC curve (AUC = {roc_auc_gan:.2f})')

    # Plot ROC curve for autoencoder output
    plt.plot(fpr_ae, tpr_ae, color='green', lw=2, label=f'Autoencoder ROC curve (AUC = {roc_auc_ae:.2f})')

    # Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # Set plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    # Save the plot as .png image
    plt.savefig('./results/pics/roc_curve.png')

    # Show the plot
    plt.show()


parser = argparse.ArgumentParser(
        description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--embedding-size', type=int, default=32, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--results_path', type=str, default='results/', metavar='N',
                    help='Where to store images')
parser.add_argument('--model', type=str, default='AE', metavar='N',
                    help='Which architecture to use')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='N',
                    help='Which dataset to use')
parser.add_argument('--loss_term', type=str, default='MSE', metavar='N',
                    help='Which loss to use')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print("device-------------:",device)
torch.manual_seed(args.seed)

# vae = VAE(args)
ae = AE(args)
ae_384 = AE_384(args)
unet = UNet(args)
ae_head = AE_head(args)
ae_gan = AE_GAN(args)

architectures = {'AE':  ae,
                 'AE_384': ae_384,
                 'UNet': unet,
                 'AE_head': ae_head,
                 'AE_GAN': ae_gan
                 }

print(args.model)
if __name__ == "__main__":
    try:
        os.stat(args.results_path)
    except :
        os.mkdir(args.results_path)

    try:
        autoenc = architectures[args.model]
    except KeyError:
        print('---------------------------------------------------------')
        print('Model architecture not supported. ', end='')
        print('Maybe you can implement it?')
        print('---------------------------------------------------------')

    autoenc = AE_GAN(args)
    ae_384 = AE_384(args)

    # Load the saved state dictionaries of the generator and discriminator models
    autoenc.generator.load_state_dict(torch.load('./saved_models/best_generator.pth'))
    autoenc.discriminator.load_state_dict(torch.load('./saved_models/best_discriminator.pth'))
    ae_384.model.load_state_dict(torch.load('./saved_models/best_AE_384.pth'))

    ## load the dataloader 
    train_loader = create_data_loaders('./datasets/size_384/normal/all/train')
    test_loader = create_data_loaders('./datasets/size_384/normal/all/test')
    abnormal_loader = create_data_loaders('./datasets/size_384/abnormal/train')

    # Set models to evaluation mode
    autoenc.generator.eval()
    autoenc.discriminator.eval()
    ae_384.model.eval()

    ## print the accuaracy and the roc curve 
    acu_roc(autoenc,ae_384, train_loader)
