import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    trainset = torchvision.datasets.CIFAR10(root='./CNN/datasets',
                                            train=True,
                                            download=True,
                                            transform=transforms.ToTensor())

    print(trainset)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=8,
                                              shuffle=True,
                                              num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./CNN/datasets',
                                           train=False,
                                           download=True,
                                           transform=transforms.ToTensor())

    print(testset)

    testloader = torch.utils.data.DataLoader(trainset,
                                             batch_size=8,
                                             shuffle=False,
                                             num_workers=2)

    labels = ('plane', 'car', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck')

    # images_batch, labels_batch = iter(trainloader).next()
    # for images_batch, labels_batch in trainloader:
        # Process the batch of data
    data_iter = iter(trainloader)
    images_batch, labels_batch = next(data_iter)

    print(images_batch.shape)

    img = torchvision.utils.make_grid(images_batch)
    print(img.shape)

    print(np.transpose(img, (1,2,0)).shape)

    plt.imshow(np.transpose(img, (1,2,0)))
    plt.axis('off')
    plt.show()

    #NN Const
    in_size = 3
    hid1_size = 16
    hid2_size = 32
    out_size = len(labels)
    k_conv_size = 5

    class ConvNet(nn.Module):
        def __init__(self, in_size, hid1_size, hid2_size, k_conv_size, out_size):
            super(ConvNet, self).__init__()

            self.layer1 = nn.Sequential(
                nn.Conv2d(in_size, hid1_size, kernel_size=k_conv_size),
                nn.BatchNorm2d(hid1_size),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2))

            self.layer2 = nn.Sequential(
                nn.Conv2d(hid1_size, hid2_size, kernel_size=k_conv_size),
                nn.BatchNorm2d(hid2_size),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2))

            self.fc = nn.Linear(hid2_size * k_conv_size * k_conv_size, out_size)

        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            # out = self.relu(self.conv1(out))
            out = out.view(out.size(0), -1)
            out = self.fc(out)
            return out

    model = ConvNet(in_size, hid1_size, hid2_size, k_conv_size, out_size)
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)

    total_step = len(trainloader)
    num_epochs = 5

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 2000 == 0:
                print('Epoch [{}/{}], Steps [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, i, total_step, loss.item()))

    # Evaluating the Model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model on the 10000 test images: {}%'\
              .format(100 * correct / total))
