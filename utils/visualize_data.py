import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def visualize_data(visualize=True, save=False):

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    if os.path.exists("../data") == False:
        os.mkdir("../data")

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    

    for idx, data in enumerate(trainloader):
        images, labels = data
        print(images.shape)
        
        # print labels
        print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

        # show images
        if visualize:
            imshow(torchvision.utils.make_grid(images)) # input torch.Size([4, 3, 32, 32]) Make a grid of images.

        if save:
            torchvision.utils.save_image(images, f'../data/{idx}_image.png')
            print(f"Image saved at ./data/{idx}_image.png")
   

def imshow(img): # img input torch.Size([4, 3, 32, 32]) -> # Function to show an image
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Visualize data')
    parser.add_argument('--visualize', action='store_true', help='Visualize data')
    parser.add_argument('--save', action='store_true', help='Save data')
    
    args = parser.parse_args()

    # Visualize data
    visualize_data(visualize=args.visualize, save=args.save)
