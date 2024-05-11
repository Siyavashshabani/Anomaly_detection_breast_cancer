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


from utils import get_max_error, plot_hist



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
    train_loader = create_data_loaders('./datasets/size_384/normal/all/train')
    test_loader = create_data_loaders('./datasets/size_384/normal/all/test')
    abnormal_loader = create_data_loaders('./datasets/size_384/abnormal/train')

    # Set models to evaluation mode
    autoenc.generator.eval()
    autoenc.discriminator.eval()
    ae_384.model.eval()


    ## create fake data 
    # fake_data = pass_data_through_models(train_loader, autoenc.generator, device, data_type="dataloader")

    # Assume `original_images` and `generated_images` are your tensors of [batch_size, channels, height, width]
    # mse_loss = F.mse_loss(fake_data, train_loader, reduction='mean')
    # max_error = (original_images - generated_images).abs().max()


    max_errors_normal= get_max_error(train_loader, autoenc.generator, device)
    max_errors_normal = torch.cat([t.view(-1) for t in max_errors_normal])
    print("average of trainloader",torch.mean(max_errors_normal), torch.std(max_errors_normal)/np.sqrt(len(max_errors_normal)) )

    max_errors_abnormal= get_max_error(abnormal_loader, autoenc.generator, device)
    max_errors_abnormal = torch.cat([t.view(-1) for t in max_errors_abnormal])
    print("average of abnormal loader",torch.mean(max_errors_abnormal), torch.std(max_errors_abnormal)/np.sqrt(len(max_errors_normal)) )

    ## plot the maximum error 
    print("type(max_errors_abnormal):",type(max_errors_abnormal) )
    plot_hist(max_errors_normal,max_errors_abnormal)
## python maximum_error.py --model AE_GAN
