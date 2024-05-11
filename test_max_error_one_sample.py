import numpy as np 

import matplotlib.pyplot as plt 
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
from utils import get_interpolations
from utils import loss_plot, plots
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
from utils import get_interpolations
from utils import loss_plot, plots
import random
from dataloaders import create_data_loaders
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


from utils import get_max_error, plot_hist, plot_sample_error






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

# args = parser.parse_args()
# args.cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if args.cuda else "cpu")
# print("device-------------:",device)


args = parser.parse_args()
# Checking CUDA availability and setting the device
if not args.no_cuda and torch.cuda.is_available():
    torch.cuda.set_device(1)  # GPU number 2 (index 1)
    device = torch.device("cuda:1")
    print(f"Using GPU: {torch.cuda.get_device_name(1)}")

args.cuda = device
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
    autoenc.generator.load_state_dict(torch.load('./saved_models/best_generator_100epchs.pth'))
    autoenc.discriminator.load_state_dict(torch.load('./saved_models/best_discriminator_100epchs.pth'))
    ae_384.model.load_state_dict(torch.load('./saved_models/best_AE_384.pth'))

    ## load the dataloader 
    # train_loader = create_data_loaders('./datasets/size_384/normal/all/train')
    # test_loader = create_data_loaders('./datasets/size_384/normal/all/test')
    # abnormal_loader = create_data_loaders('./datasets/size_384/abnormal/train')

    # Set models to evaluation mode
    autoenc.generator.eval()
    autoenc.discriminator.eval()
    ae_384.model.eval()




    img = np.load("./data/processed/abnormal/img/1_4030.npy")
    gt = np.load("./data/processed/abnormal/mask/1_4030.npy")
    print(img.shape) 
    img = torch.from_numpy(img).float().to(device).unsqueeze(0).unsqueeze(0)
    print(img.shape) 
    with torch.no_grad():
        outputs = autoenc.generator(img)
    # mse_loss = F.mse_loss(outputs, data, reduction='mean')
    max_error = (outputs - img).abs()

    ## calculate the max error for abnormal pixels 
    abnormal_pixels = max_error.cpu().numpy()*gt
    non_zero_pixels = abnormal_pixels[abnormal_pixels != 0]
    mean_non_zero_pixels = np.mean(non_zero_pixels)
    print("mean of abnormal pixels:",mean_non_zero_pixels)

    ## calculate the max error for normal pixels 
    inverted_gt = np.logical_not(gt).astype(int)
    abnormal_pixels = max_error.cpu().numpy()*inverted_gt
    zero_pixels = abnormal_pixels[abnormal_pixels != 0]
    mean_zero_pixels = np.mean(zero_pixels)
    print("mean of normal pixels:",mean_zero_pixels )


    plot_sample_error(img[0,0,:,:].cpu().numpy(),outputs[0,0,:,:].cpu().numpy(), max_error[0,0,:,:].cpu().numpy(), gt,mean_non_zero_pixels, mean_zero_pixels)
