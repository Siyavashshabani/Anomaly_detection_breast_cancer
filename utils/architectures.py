import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as  np

class FC_Encoder(nn.Module):
    def __init__(self, output_size):
        super(FC_Encoder, self).__init__()
        self.fc1 = nn.Linear(784, output_size)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        return h1

class FC_Decoder(nn.Module):
    def __init__(self, embedding_size):
        super(FC_Decoder, self).__init__()
        self.fc3 = nn.Linear(embedding_size, 1024)
        self.fc4 = nn.Linear(1024, 784)

    def forward(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_size=(1, 128, 128)):
        super(CNN_Encoder, self).__init__()

        self.input_size = input_size
        self.channel_mult = 16

        # Define each convolutional layer and activation layer explicitly
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.channel_mult*1, kernel_size=4, stride=1, padding=1)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(self.channel_mult*1, self.channel_mult*2, 4, 2, 1)
        self.bn2 = nn.BatchNorm2d(self.channel_mult*2)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(self.channel_mult*2, self.channel_mult*4, 4, 2, 1)
        self.bn3 = nn.BatchNorm2d(self.channel_mult*4)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv4 = nn.Conv2d(self.channel_mult*4, self.channel_mult*8, 4, 2, 1)
        self.bn4 = nn.BatchNorm2d(self.channel_mult*8)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)
        self.conv5 = nn.Conv2d(self.channel_mult*8, self.channel_mult*16, 3, 2, 1)
        self.bn5 = nn.BatchNorm2d(self.channel_mult*16)
        self.lrelu5 = nn.LeakyReLU(0.2, inplace=True)

        # self.flat_fts = self.get_flat_fts()

        # self.linear = nn.Sequential(
        #     nn.Linear(self.flat_fts, output_size),
        #     nn.BatchNorm1d(output_size),
        #     nn.LeakyReLU(0.2),
        # )

    # def get_flat_fts(self):
    #     f = fts(Variable(torch.ones(1, *self.input_size)))
    #     return int(np.prod(f.size()[1:]))
   
    def forward(self, x):
        print("input of forward path: ",x.shape)
        x = self.lrelu1(self.conv1(x))
        print("output of first conv layer: ",x.shape)
        x = self.lrelu2(self.bn2(self.conv2(x)))
        x = self.lrelu3(self.bn3(self.conv3(x)))
        x = self.lrelu4(self.bn4(self.conv4(x)))
        x = self.lrelu5(self.bn5(self.conv5(x)))
        x = x.view(-1, self.flat_fts)
        x = self.lrelu_fc(self.bn_fc(self.fc(x)))
        return x

class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_size=(1, 28, 28)):
        super(CNN_Decoder, self).__init__()
        self.input_height = 28
        self.input_width = 28
        self.input_dim = embedding_size
        self.channel_mult = 16
        self.output_channels = 1
        self.fc_output_dim = 512

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, self.fc_output_dim),
            nn.BatchNorm1d(self.fc_output_dim),
            nn.ReLU(True)
        )

        self.deconv = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.fc_output_dim, self.channel_mult*4,
                                4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_mult*4),
            nn.ReLU(True),
            # state size. self.channel_mult*32 x 4 x 4
            nn.ConvTranspose2d(self.channel_mult*4, self.channel_mult*2,
                                3, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*2),
            nn.ReLU(True),
            # state size. self.channel_mult*16 x 7 x 7
            nn.ConvTranspose2d(self.channel_mult*2, self.channel_mult*1,
                                4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_mult*1),
            nn.ReLU(True),
            # state size. self.channel_mult*8 x 14 x 14
            nn.ConvTranspose2d(self.channel_mult*1, self.output_channels, 4, 2, 1, bias=False),
            nn.Sigmoid()
            # state size. self.output_channels x 28 x 28
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, self.fc_output_dim, 1, 1)
        x = self.deconv(x)
        return x.view(-1, self.input_width*self.input_height)





class Encoder(nn.Module):
    def __init__(self, image_size, channels, embedding_dim):
        super(Encoder, self).__init__()
        # define convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        # variable to store the shape of the output tensor before flattening
        # the features, it will be used in decoders input while reconstructing
        self.shape_before_flattening = None
        # compute the flattened size after convolutions
        flattened_size = (image_size // 8) * (image_size // 8) * 128
        # define fully connected layer to create embeddings
        self.fc = nn.Linear(flattened_size, embedding_dim)
    def forward(self, x):
        # apply ReLU activations after each convolutional layer
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # store the shape before flattening
        # print("x before flatting--------------------------------------------------", x.shape)
        self.shape_before_flattening = x.shape[1:]
        # print("self.shape_before_flattening ..............", self.shape_before_flattening )
        # print("np.prod(shape_before_flattening)------------", np.prod(self.shape_before_flattening))
        # flatten the tensor
        x = x.view(x.size(0), -1)
        # apply fully connected layer to generate embeddings
        # print("shape after flatting", x.shape)
        x = self.fc(x)
        # print("shape after fully connected layer:", x.shape)
        return x
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim, shape_before_flattening, channels):
        super(Decoder, self).__init__()
        # define fully connected layer to unflatten the embeddings
        self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flattening))
        # store the shape before flattening
        self.reshape_dim = shape_before_flattening
        # define transpose convolutional layers
        self.deconv1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # define final convolutional layer to generate output image
        self.conv1 = nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # print("start decoder block -----------------------------:")
        # apply fully connected layer to unflatten the embeddings
        x = self.fc(x)
        # reshape the tensor to match shape before flattening
        x = x.view(x.size(0), *self.reshape_dim)
        # apply ReLU activations after each transpose convolutional layer
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        # apply sigmoid activation to the final convolutional layer to generate output image
        x = torch.sigmoid(self.conv1(x))
        # print("output shape of decodr: ", x.shape)
        return x




###################### removing the stride and maxpooling 

class Encoder_maxpool(nn.Module):
    def __init__(self, image_size, channels, embedding_dim):
        super(Encoder_maxpool, self).__init__()
        # define convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding = 1)
        self.pool1= nn.MaxPool2d(2,stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding = 1)
        self.pool2= nn.MaxPool2d(2,stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding = 1)
        self.pool3= nn.MaxPool2d(2, stride=2)
        # variable to store the shape of the output tensor before flattening
        # the features, it will be used in decoders input while reconstructing
        self.shape_before_flattening = None
        # compute the flattened size after convolutions
        flattened_size = (image_size // 8) * (image_size // 8) * 128
        # define fully connected layer to create embeddings
        self.fc = nn.Linear(flattened_size, embedding_dim)
    def forward(self, x):
        # apply ReLU activations after each convolutional layer
        x = F.relu(self.conv1(x))
        # print("after first conv", x.shape)
        x = self.pool1(x)
        # print("first maxpooling:", x.shape)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # print("second maxpooling:", x.shape)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        # print("third maxpooling", x.shape)
        # store the shape before flattening
        # print("x before flatting--------------------------------------------------", x.shape)
        self.shape_before_flattening = x.shape[1:]
        print("self.shape_before_flattening.shape-----------:", self.shape_before_flattening)
        # print("self.shape_before_flattening ..............", self.shape_before_flattening )
        # print("np.prod(shape_before_flattening)------------", np.prod(self.shape_before_flattening))
        # flatten the tensor
        x = x.view(x.size(0), -1)
        # apply fully connected layer to generate embeddings
        # print("shape after flatting", x.shape)
        x = self.fc(x)
        # print("shape after fully connected layer:", x.shape)
        return x
    


########################### UNet 

from torch import nn
import torch 
from torch.nn import functional as F


class UNet(nn.Module):
    def __init__(self, n_class= 1):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = F.relu(self.e11(x))
        xe12 = F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.e21(xp1))
        xe22 = F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.e31(xp2))
        xe32 = F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.e41(xp3))
        xe42 = F.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.d41(xu44))
        xd42 = F.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out
    

 ######################### encoder and decoder  for input that has input shape as 384

class Encoder_384(nn.Module):
    def __init__(self, image_size, channels, embedding_dim):
        super(Encoder_384, self).__init__()
        # define convolutional layers
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, padding = 1)
        self.pool1= nn.MaxPool2d(2,stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding = 1)
        self.pool2= nn.MaxPool2d(2,stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding = 1)
        self.pool3= nn.MaxPool2d(2, stride=2)

        # self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding = 1)
        # self.pool4= nn.MaxPool2d(2, stride=2)        
        # variable to store the shape of the output tensor before flattening
        # the features, it will be used in decoders input while reconstructing
        self.shape_before_flattening = None
        # compute the flattened size after convolutions
        flattened_size = (image_size // 8) * (image_size // 8) * 128
        # define fully connected layer to create embeddings
        self.fc1 = nn.Linear(flattened_size, 100)
        self.fc2 = nn.Linear(100, embedding_dim)
    def forward(self, x):
        # apply ReLU activations after each convolutional layer
        x = F.relu(self.conv1(x))
        # print("after first conv", x.shape)
        x = self.pool1(x)
        # print("first maxpooling:", x.shape)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # print("second maxpooling:", x.shape)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        # print("third maxpooling", x.shape)
        # store the shape before flattening
        # print("x before flatting--------------------------------------------------", x.shape)
        self.shape_before_flattening = x.shape[1:]
        # print("self.shape_before_flattening ..............", self.shape_before_flattening )
        # print("np.prod(shape_before_flattening)------------", np.prod(self.shape_before_flattening))
        # flatten the tensor
        x = x.view(x.size(0), -1)
        # apply fully connected layer to generate embeddings
        # print("shape after flatting", x.shape)
        x = self.fc2(self.fc1(x) ) 
        # print("shape after fully connected layer:", x.shape)
        return x


class Decoder_384(nn.Module):
    def __init__(self, embedding_dim, shape_before_flattening, channels):
        super(Decoder_384, self).__init__()
        # define fully connected layer to unflatten the embeddings
        self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flattening))
        # store the shape before flattening
        self.reshape_dim = shape_before_flattening
        # define transpose convolutional layers
        self.deconv1 = nn.ConvTranspose2d(
            128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv2 = nn.ConvTranspose2d(
            128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.deconv3 = nn.ConvTranspose2d(
            64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        # define final convolutional layer to generate output image
        self.conv1 = nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        # print("start decoder block -----------------------------:")
        # apply fully connected layer to unflatten the embeddings
        x = self.fc(x)
        # reshape the tensor to match shape before flattening
        x = x.view(x.size(0), *self.reshape_dim)
        # apply ReLU activations after each transpose convolutional layer
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        # apply sigmoid activation to the final convolutional layer to generate output image
        x = torch.sigmoid(self.conv1(x))
        # print("output shape of decodr: ", x.shape)
        return x



class Discriminator(nn.Module):
    def __init__(self, channels= 1, input_size = 384):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),  # Pooling to reduce the dimensions
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),  # Further reduction in dimensions
            nn.Flatten(),
            nn.Linear(16 * int(input_size/4) * int(input_size/4), 1),  # Adjust size according to output of last MaxPool
        )
        self.output = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.network(x)
        return self.output(x)
    

class Discriminator_shallow(nn.Module):
    def __init__(self, channels= 1, input_size = 384):
        super(Discriminator_shallow, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),  # Pooling to reduce the dimensions
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),  # Further reduction in dimensions
            nn.Flatten(),
            nn.Linear(32 * int(input_size/4) * int(input_size/4), 1),  # Adjust size according to output of last MaxPool
        )
        self.output = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.network(x)
        return self.output(x)

