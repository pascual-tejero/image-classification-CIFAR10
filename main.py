import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn

import os

from net import Net

PATH = './cifar_net.pth' # Path to save the trained model

def main():
    # Set the environment variable to allow the use of the MKL library
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    # Transform the data to tensors and normalize it    
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ) 

    batch_size = 4 # Batch size

    if os.path.exists("./data") == False: # Check if the data directory exists
        os.mkdir("./data") # Create the data directory

    # Download the CIFAR10 training dataset.
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    
    # There are 50,000 images in the training set. We will use 45,000 for training and 5,000 for validation
    trainset, valset = torch.utils.data.random_split(trainset, [45000, 5000])

    # Load the training and validation data into the memory
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2) 
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)
    
    # Download the CIFAR10 dataset and load it into the memory
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2) 

    net = Net() # Create the network
    
    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # Optimizer
    
    print("---------------------------------------")
    print('Start training')
    for epoch in range(20):  # loop over the dataset multiple times

        running_loss = 0.0 # Loss for each epoch
        print("---------------------------------------")

        # Train the network
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
        
        # Validate the network
        running_loss = 0.0
        for i, data in enumerate(valloader, 0):
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        
        print(f'Epoch: {epoch + 1}, Validation Loss: {running_loss / len(valloader):.3f}')

        # Save the model with the lowest validation loss
        if i == len(valloader) - 1:
            if epoch == 0:
                lowest_loss = running_loss # Set the lowest loss to the first loss
            elif running_loss < lowest_loss:
                lowest_loss = running_loss
                torch.save(net.state_dict(), PATH) # Save the trained model

    print('Finished Training')
    
    # Test the network on the test data
    print("---------------------------------------")
    print('Start testing')

    # Load the best model
    net.load_state_dict(torch.load(PATH))

    tp = 0 # True Positive
    tn = 0 # True Negative
    fp = 0 # False Positive
    fn = 0 # False Negative

    for data in testloader:
        images, labels = data # Get the images and labels from the test data
        outputs = net(images) # Get the output from the network
        _, predicted = torch.max(outputs.data, 1) # Get the class with the highest probability
        tp += ((predicted == labels) & (predicted == 1)).sum().item() # True Positive
        tn += ((predicted == labels) & (predicted == 0)).sum().item() # True Negative 
        fp += ((predicted != labels) & (predicted == 1)).sum().item() # False Positive
        fn += ((predicted != labels) & (predicted == 0)).sum().item() # False Negative

    # Print the results    
    print("---------------------------------------")
    print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}') 
    print(f'Accuracy: {(tp + tn) / (tp + tn + fp + fn):.3f}')
    print(f'Precision: {tp / (tp + fp):.3f}')
    print(f'Recall: {tp / (tp + fn):.3f}')
    print(f'Specificity: {tn / (tn + fp):.3f}')
    print(f'F1: {tp / (tp + (fp + fn) / 2):.3f}')

    print('Finished Testing')
    print("---------------------------------------")

if __name__ == '__main__':
    main()