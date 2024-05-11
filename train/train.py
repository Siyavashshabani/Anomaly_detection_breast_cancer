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
from models.AE_head import AE_head
import matplotlib.pyplot as plt 
from utils import get_interpolations
from utils import loss_plot
import random
from utils import loss_plot, plots_AE


parser = argparse.ArgumentParser(
        description='Main function to call training for different AutoEncoders')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
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
torch.manual_seed(args.seed)

# vae = VAE(args)
ae = AE(args)
ae_384 = AE_384(args)
unet = UNet(args)
ae_head = AE_head(args)
architectures = {'AE':  ae,
                 'AE_384': ae_384,
                 'aUNet': unet,
                 'AE_head': ae_head}

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
        sys.exit()
    if args.loss_term == "MSE":
        loss_function = nn.MSELoss()
    elif args.loss_term == "MAE":
        loss_function = nn.L1Loss()  # MAE is equivalent to L1 Loss
    elif args.loss_term == "RMSE":
        class RMSELoss(nn.Module):
            def __init__(self):
                super(RMSELoss, self).__init__()

            def forward(self, predicted, actual):
                return torch.sqrt(nn.MSELoss()(predicted, actual))

        loss_function = RMSELoss() 
    try:
        train_losses = []
        test_losses = []
        best_loss = float('inf')
        for epoch in range(1, args.epochs + 1):
            train_loss = autoenc.train(epoch, loss_function)
            test_loss = autoenc.test(epoch, loss_function)
            train_losses.append(train_loss)
            test_losses.append(test_loss)   
            ## print the loss of three datasets and save the output of model 
            if epoch % 2 == 0:
                abnormal_loss_avg = plots_AE(autoenc, epoch, loss_function)

                ## save the model
                if abnormal_loss_avg < best_loss:
                    # Update best loss and save the model weights
                    best_loss = abnormal_loss_avg            
                    torch.save(autoenc.model.state_dict(), './saved_models/best_AE_384.pth')
                    print("-------------------flag-------------------")
    except (KeyboardInterrupt, SystemExit):
        print("Manual Interruption")

    with torch.no_grad():
        images, _ = next(iter(autoenc.test_loader))
        images = images.float().to(autoenc.device)
        images_per_row = random.randint(0, 31)
        # interpolations = get_interpolations(args, autoenc.model, autoenc.device, images, images_per_row)

        # sample = torch.randn(64, args.embedding_size).to(autoenc.device)
        # sample = autoenc.model.decode(sample).cpu()
        # save_image(sample.view(64, 1, 28, 28),
        #         '{}/sample_{}_{}.png'.format(args.results_path, args.model, args.dataset))

        ## Save test data 
        pred = autoenc.model(images)
        np.save("./results/test_input.npy",images[images_per_row,0,:,:].cpu())
        np.save("./results/test_output.npy",pred[images_per_row,0,:,:].cpu())

        ## ploting the curve for loss of test and train
        loss_plot(train_losses,test_losses )     

        ## calculate the distribution of loss values 
        train_loss_avg = autoenc.loss_total(autoenc.train_loader, loss_function)
        test_loss_avg = autoenc.loss_total(autoenc.test_loader, loss_function)
        abnormal_loss_avg = autoenc.loss_total(autoenc.abnormal_loader, loss_function)

        print("image name-------------------:", images_per_row)
        print("train loss average-------------:", train_loss_avg)
        print("test loss average-------------:", test_loss_avg)
        print("Abnormal loss average-------------:", abnormal_loss_avg)


        # save_image(interpolations.view(-1, 1, 28, 28),
        #         '{}/interpolations_{}_{}.png'.format(args.results_path, args.model, args.dataset),  nrow=images_per_row)
        # interpolations = interpolations.cpu()
        # interpolations = np.reshape(interpolations.data.numpy(), (-1, 28, 28))
        # interpolations = ndimage.zoom(interpolations, 5, order=1)
        # interpolations *= 256
        # imageio.mimsave('{}/animation_{}_{}.gif'.format(args.results_path, args.model, args.dataset), interpolations.astype(np.uint8))
