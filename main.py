import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn


import os

from net import Net

def main():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    if os.path.exists("./data") == False:
        os.mkdir("./data")

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    net = Net()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    print('Start training')
    for epoch in range(5):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
                  
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'Epoch: {epoch + 1}, Batch: {i + 1:5d} -> Loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # Test the network on the test data
    overall_loss = 0.0
    for data in testloader:
        images, labels = data
        outputs = net(images)
        loss = criterion(outputs, labels)
        overall_loss += loss.item()
    print(f'Overall loss test: {overall_loss / len(testloader):.3f}')


if __name__ == '__main__':
    main()