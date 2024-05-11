import torch
import torch.utils.data
import torch.nn as nn
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

import sys
sys.path.append('../')
from utils.architectures import FC_Encoder, FC_Decoder, CNN_Encoder, CNN_Decoder, Encoder, Decoder, Encoder_maxpool
from dataloaders.datasets import MNIST, EMNIST, FashionMNIST
from dataloaders.dataloaders import create_data_loaders


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        output_size = args.embedding_size
        self.encoder = Encoder_maxpool(image_size= 128, channels= 1, embedding_dim= output_size)
        self.classifier = nn.Linear(output_size, 2)
        self.decoder = Decoder(embedding_dim= output_size, shape_before_flattening = [128,16,16], channels=1)

    def encode(self, x):
        z = self.encoder(x)
        head = torch.sigmoid(self.classifier(z) )
        return z, head 

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # print("shape of input data", x.shape)
        # print("after changing the view", x.view(-1, 524288).shape )
        z, head = self.encode(x)
        output = self.decode(z)
        return output, head

class AE_head(object):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        # self._init_dataset()
        self.train_loader = create_data_loaders('./datasets/size_384/normal/low/train')
        self.test_loader = create_data_loaders('./datasets/size_384/normal/low/test')
        self.abnormal_loader = create_data_loaders('./datasets/size_384/abnormal/test')
        self.model = Network(args)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

  
    def train(self, epoch, loss_function, optimizer):
        self.model.train()
        BCELoss_function = nn.BCELoss()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(self.train_loader):
            data = data.float().to(self.device)
            optimizer.zero_grad()
            # print("shape of input:", data.shape)
            recon_batch, pred_cls = self.model(data)

            if epoch % 2 == 0:
                loss = loss_function(recon_batch, data)

            else:
                joint_data = torch.cat((data, recon_batch), dim=0)

                joint_pred, pred_cls = self.model(joint_data)
                classes = torch.zeros(2*recon_batch.shape[0], 2).to(self.device)
                classes[:recon_batch.shape[0],0] = 1
                classes[recon_batch.shape[0]:,1] = 1
                # loss1 = loss_function(joint_pred, joint_data)

                # print("pred_cls.shape-------------:",pred_cls.shape)
                # print("classes.shape-------------:",classes.shape)
                loss2 = BCELoss_function(pred_cls ,classes)
                loss = loss2 ##loss1 +
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
                recon_batch,_ = self.model(data)
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
                    recon_batch,_ = self.model(data)
                    total_loss += loss_total(recon_batch, data).item()

            total_loss /= len(loader.dataset)
            print('====> Test set loss: {:.6f}'.format(total_loss))
            return total_loss
