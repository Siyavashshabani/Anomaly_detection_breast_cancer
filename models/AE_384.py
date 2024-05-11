import torch
import torch.utils.data
import torch.nn as nn
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

import sys
sys.path.append('../')
from architectures import FC_Encoder, FC_Decoder, CNN_Encoder, CNN_Decoder, Encoder, Decoder, Encoder_maxpool, Encoder_384, Decoder_384
from datasets import MNIST, EMNIST, FashionMNIST
from dataloaders import create_data_loaders


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
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

class AE_384(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        # self._init_dataset()
        self.train_loader = create_data_loaders('./datasets/size_384/normal/all/train')
        self.test_loader = create_data_loaders('./datasets/size_384/normal/all/test')
        self.abnormal_loader = create_data_loaders('./datasets/size_384/abnormal/train')
        self.model = Network(args)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

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

    def train(self, epoch, loss_function):
        self.model.train()
        train_loss = 0

        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.float().to(self.device)
            self.optimizer.zero_grad()
            recon_batch = self.model(data)
            loss = loss_function(recon_batch, data)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.6f}'.format(
              epoch, train_loss / len(self.train_loader.dataset)))
        return train_loss/len(self.train_loader.dataset)

    def test(self, epoch, loss_function):
        self.model.eval()
        # loss_function = nn.MSELoss()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(self.test_loader):
                data = data.float().to(self.device)
                recon_batch = self.model(data)
                test_loss += loss_function(recon_batch, data).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.6f}'.format(test_loss))
        return test_loss


    def loss_total(self, loader, loss_total):
            self.model.eval()
            # mse_loss_function = nn.MSELoss()
            total_loss = 0
            with torch.no_grad():
                for i, (data, _) in enumerate(loader):
                    data = data.float().to(self.device)
                    recon_batch = self.model(data)
                    total_loss += loss_total(recon_batch, data).item()

            total_loss /= len(loader.dataset)
            print('====> Test set loss: {:.6f}'.format(total_loss))
            return total_loss
