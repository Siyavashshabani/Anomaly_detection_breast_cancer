import torch
import torch.utils.data
import torch.nn as nn
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torch.optim as optim

import sys
sys.path.append('../')
from utils.architectures import FC_Encoder, FC_Decoder, CNN_Encoder, CNN_Decoder, Encoder, Decoder, Encoder_maxpool, Encoder_384, Decoder_384
from dataloaders.datasets import MNIST, EMNIST, FashionMNIST
from dataloaders.dataloaders import create_data_loaders
from utils.architectures import Discriminator



class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        output_size = args.embedding_size
        self.encoder = Encoder_384(image_size= 384, channels= 1, embedding_dim= output_size)

        self.decoder = Decoder_384(embedding_dim= output_size, shape_before_flattening = [128,48,48], channels=1)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # print("shape of input data", x.shape)
        # print("after changing the view", x.view(-1, 524288).shape )
        z = self.encode(x)
        return self.decode(z)

class AE_GAN(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        # self._init_dataset()
        self.train_loader = create_data_loaders('./datasets/size_384/normal/all/train')
        self.test_loader = create_data_loaders('./datasets/size_384/normal/all/test')
        self.abnormal_loader = create_data_loaders('./datasets/size_384/abnormal/train')
        self.generator = Generator(args)
        self.generator.to(self.device)
        self.discriminator = Discriminator()
        self.discriminator.to(self.device)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)


        # Assuming generator and discriminator are defined as separate modules
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=5e-5, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))


    # def _init_dataset(self):
    #     if self.args.dataset == 'MNIST':
    #         self.data = MNIST(self.args)
    #     elif self.args.dataset == 'EMNIST':
    #         self.data = EMNIST(self.args)
    #     elif self.args.dataset == 'FashionMNIST':
    #         self.data = FashionMNIST(self.args)
    #     else:
    #         print("Dataset not supported")
    #         sys.exit()

    # def loss_function(self, recon_x, x):
    #     mse_loss = nn.MSELoss()
    #     loss = mse_loss(recon_x, x)
    #     return mse_loss

    def train(self, epoch, loss_function_G, loss_function_D):
        # self.model.train()
        G_loss_total = 0
        D_loss_total = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            print("self.device--------------------:",self.device)
            data = data.float().to(self.device)

            fake_images = self.generator(data.detach()).detach()  # Detach to avoid training the generator now

            # Concatenate real and fake images and labels
            combined_images = torch.cat([data, fake_images], dim=0)
            # Real images targets (class index 0 for real), fake images targets (class index 1 for fake)
            real_targets = torch.ones(data.size(0), dtype=torch.float).to(self.device).unsqueeze(1)
            fake_targets = torch.zeros(fake_images.size(0), dtype=torch.float).to(self.device).unsqueeze(1)
            combined_targets = torch.cat([real_targets, fake_targets], dim=0)
            # combined_targets = combined_targets.unsqueeze(1)

            # Optional: shuffle the combined dataset to prevent the discriminator from learning the order
            indices = torch.randperm(combined_images.size(0))
            combined_images = combined_images[indices]
            combined_targets = combined_targets[indices]

            # Update Discriminator
            self.optimizer_D.zero_grad()
            predictions = self.discriminator(combined_images)
            D_loss = loss_function_D(predictions, combined_targets)
            D_loss.backward()
            self.optimizer_D.step()
            D_loss = 0 
            
            # Update Generator (Autoencoder)
            self.optimizer_G.zero_grad()
            fake_images = self.generator(data)  # Get fresh regeneration for backprop
            predictions = self.discriminator(fake_images)
            D_loss = loss_function_D(predictions, real_targets)  # Trick discriminator
            reconstruction_loss = 500*loss_function_G(fake_images, data)
            G_loss = D_loss + reconstruction_loss
            G_loss.backward()
            self.optimizer_G.step()

            G_loss_total += G_loss.item()
            D_loss_total += D_loss.item()


            if batch_idx % self.args.log_interval == 0: ##  
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss_generator: {:.6f} \tLoss_discriminator: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset), 
                    100 * batch_idx / len(self.train_loader),
                    reconstruction_loss.item() / len(data),
                    D_loss.item() / len(data)))

        print('====> Epoch: {} Average loss G: {:.6f} Average loss D: {:.6f}'.format(
              epoch, G_loss_total / len(self.train_loader.dataset), D_loss_total / len(self.train_loader.dataset)))

        
        return G_loss_total/len(self.train_loader.dataset)

    def test(self, epoch, loss_function):
        self.generator.eval()
        # loss_function = nn.MSELoss()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.float().to(self.device)
                recon_batch = self.generator(data)
                test_loss += loss_function(recon_batch, data).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.6f}'.format(test_loss))
        return test_loss


    def loss_total(self, loader, loss_total):
            self.generator.eval()
            # mse_loss_function = nn.MSELoss()
            total_loss = 0
            with torch.no_grad():
                for i, (data, _) in enumerate(loader):
                    data = data.float().to(self.device)
                    recon_batch = self.generator(data)
                    total_loss += loss_total(recon_batch, data).item()

            total_loss /= len(loader.dataset)
            print('====> Test set loss: {:.6f}'.format(total_loss))
            return total_loss
