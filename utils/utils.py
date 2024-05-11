import numpy as np
import torch
import matplotlib.pyplot as plt 
import random 
def loss_plot(train_losses,test_losses ):

    train_losses_np = [tensor for tensor in train_losses]
    test_losses_np = [tensor for tensor in test_losses]

    plt.figure(figsize=(10, 5))
    plt.plot(np.array(train_losses_np), label='Train Loss')
    plt.plot(np.array(test_losses_np), label='Test Loss')
    plt.title('Loss Progression')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/loss_curve.png')
    plt.show()

def get_interpolations(args, model, device, images, images_per_row=20):
    model.eval()
    with torch.no_grad():
        def interpolate(t1, t2, num_interps):
            alpha = np.linspace(0, 1, num_interps+2)
            interps = []
            for a in alpha:
                interps.append(a*t2.view(1, -1) + (1 - a)*t1.view(1, -1))
            return torch.cat(interps, 0)

        if args.model == 'VAE':
            mu, logvar = model.encode(images.view(-1, 784))
            embeddings = model.reparameterize(mu, logvar).cpu()
        elif args.model == 'AE':
            embeddings = model.encode(images.view(-1, 784))
            
        interps = []
        for i in range(0, images_per_row+1, 1):
            interp = interpolate(embeddings[i], embeddings[i+1], images_per_row-4)
            interp = interp.to(device)
            interp_dec = model.decode(interp)
            line = torch.cat((images[i].view(-1, 784), interp_dec, images[i+1].view(-1, 784)))
            interps.append(line)
        # Complete the loop and append the first image again
        interp = interpolate(embeddings[i+1], embeddings[0], images_per_row-4)
        interp = interp.to(device)
        interp_dec = model.decode(interp)
        line = torch.cat((images[i+1].view(-1, 784), interp_dec, images[0].view(-1, 784)))
        interps.append(line)

        interps = torch.cat(interps, 0).to(device)
    return interps


def plots(autoenc, epoch, loss_function_G):
        with torch.no_grad():
            images, _ = next(iter(autoenc.test_loader))
            images = images.float().to(autoenc.device)
            images_per_row = random.randint(0, 31)

            ## Save test data 
            pred = autoenc.generator(images)
            np.save("./results/test_input_{}.npy".format(epoch),images[images_per_row,0,:,:].cpu())
            np.save("./results/test_output_{}.npy".format(epoch),pred[images_per_row,0,:,:].cpu())

            ## calculate the distribution of loss values 
            train_loss_avg = autoenc.loss_total(autoenc.train_loader, loss_function_G)
            test_loss_avg = autoenc.loss_total(autoenc.test_loader, loss_function_G)
            abnormal_loss_avg = autoenc.loss_total(autoenc.abnormal_loader, loss_function_G)

            print("image name-------------------:", images_per_row)
            print("train loss average-------------:", train_loss_avg)
            print("test loss average-------------:", test_loss_avg)
            print("Abnormal loss average-------------:", abnormal_loss_avg)
            return abnormal_loss_avg 



def plots_AE(autoenc, epoch, loss_function):
        with torch.no_grad():
            # images, _ = next(iter(autoenc.test_loader))
            # images = images.float().to(autoenc.device)
            # images_per_row = random.randint(0, 31)

            ## Save test data 
            # pred = autoenc.model(images)
            # np.save("./results/test_input_{}.npy".format(epoch),images[images_per_row,0,:,:].cpu())
            # np.save("./results/test_output_{}.npy".format(epoch),pred[images_per_row,0,:,:].cpu())

            ## calculate the distribution of loss values 
            train_loss_avg = autoenc.loss_total(autoenc.train_loader, loss_function)
            test_loss_avg = autoenc.loss_total(autoenc.test_loader, loss_function)
            abnormal_loss_avg = autoenc.loss_total(autoenc.abnormal_loader, loss_function)

            # print("image name-------------------:", images_per_row)
            print("train loss average-------------:", train_loss_avg)
            print("test loss average-------------:", test_loss_avg)
            print("Abnormal loss average-------------:", abnormal_loss_avg)
            return abnormal_loss_avg 