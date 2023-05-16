import matplotlib.pyplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.optim as optim

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

titanic_data = pd.read_csv('../Neural_Network/datasets/tested.csv')
print(titanic_data.head())

# Trimming unwanted data
unwanted_features = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Embarked']
titanic_data = titanic_data.drop(unwanted_features, axis=1)
print(titanic_data.head())

# Removing nulls
titanic_data = titanic_data.dropna()

# Preporocessing
le = preprocessing.LabelEncoder()
titanic_data['Sex'] = le.fit_transform(titanic_data['Sex'])
print(titanic_data.head())

# Features

features = ['Pclass', 'Sex', 'Age', 'Fare']
titanic_features = titanic_data[features]
print(titanic_features.head())

# Pclass columns One-hot encoding
titanic_features = pd.get_dummies(titanic_features, columns=['Pclass'])
print(titanic_features.head())

titanic_target = titanic_data[['Survived']]

X_train, x_test, Y_train, y_test = train_test_split(titanic_features, titanic_target, test_size=0.2, random_state=0)

print(X_train.shape, Y_train.shape)

import torch

Xtrain_ = torch.from_numpy(X_train.values.astype('float32'))
Xtest_ = torch.from_numpy(x_test.values.astype('float32'))

print(Xtrain_.shape)

# Reshaping the Y train and test into the format that our loss function requires NNL loss function
Ytrain_ = torch.from_numpy(Y_train.values).view(1, -1)[0]
Ytest_ = torch.from_numpy(y_test.values).view(1, -1)[0]
print(Ytrain_.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Defining NN Constants
input_size = 6
output_size = 2
hidden_size = 10


class Net(nn.Module):

    # def __int__(self):
    #     # This allows us to initialize the NN before initializing layers
    #     super(Net, self).__init__()
    #     # Instantiating 3 Linear, fully-connected layers
    #     self.fc1 = nn.Linear(6, 10)
    #     self.fc2 = nn.Linear(10, 10)
    #     self.fc3 = nn.Linear(10, 2)

    # def forward(self, x):
    #     x = F.sigmoid(self.fc1(x))
    #     x = F.sigmoid(self.fc2(x))
    #     x = self.fc3(x)
    #
    #     return F.log_softmax(x, dim=-1)
    def log_softmax(self, x):
        return torch.log(torch.softmax(x, dim=-1))

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.sigmoid(self.layer1(x))
        x = F.sigmoid(self.layer2(x))
        x = self.layer3(x)
        x = self.log_softmax(x)
        return x


# Initializing NN Model with override 3 layer of Net class with PyTorch nn class
model = Net()

print(list(model.parameters()))
# Initialing Gradient Descent for forward pass
optimizer = optim.Adam(model.parameters())
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# Initializing Loss Function NNLLoss for the backward pass
loss_fn = nn.NLLLoss()

epoch_data = []
epochs = 1001

for epoch in range(1, epochs):

    # Forward Pass
    optimizer.zero_grad()
    Ypred = model(Xtrain_)

    # Backward pass
    loss = loss_fn(Ypred, Ytrain_)
    loss.backward()

    # Updating the model parameters
    optimizer.step()

    Ypred_test = model(Xtest_)
    loss_test = loss_fn(Ypred_test, Ytest_)

    _, pred = Ypred_test.data.max(1)

    accuracy = pred.eq(Ytest_.data).sum().item() / y_test.values.size
    epoch_data.append([epoch, loss.data.item(), loss_test.data.item(), accuracy])

    if epoch % 100 == 0:
        print('epoch - %d (%d%%) train loss - %.2f test loss - %.2f accuracy - %.4f' \
            % (epoch, epoch / 150 * 10, loss.data.item(), loss_test.data.item(), accuracy))

df_epochs_data = pd.DataFrame(epoch_data, columns=['epoch', 'train_loss', 'test_loss', 'accuracy'])
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
df_epochs_data[['train_loss', 'test_loss']].plot(ax=ax1)
df_epochs_data[['accuracy']].plot(ax=ax2)
plt.ylim(ymin=0.5)
plt.show()

