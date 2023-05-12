import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# automobile_data = pd.read_csv('..\ComputerVisionByPytorch\datasets\Automobile_data.csv', sep=r'\s*,\S*', engine='python')
automobile_data = pd.read_csv('../Neural_Network/datasets/Automobile_data.csv')

print(automobile_data.head())
print(automobile_data.info())

automobile_data = automobile_data.replace('?', np.nan)
print(automobile_data.head())
automobile_data = automobile_data.dropna()

col = ['make', 'fuel-type', 'body-style', 'horsepower']
automobile_features = automobile_data[col]
print(automobile_features.head())

automobile_target = automobile_data[['price']]
print(automobile_target)
print(automobile_features['horsepower'].describe())

pd.options.mode.chained_assignment = None

automobile_features['horsepower'] = pd.to_numeric(automobile_features['horsepower'])

print(automobile_features['horsepower'].describe())

automobile_target = automobile_target.astype(float)
print(automobile_target['price'].describe())

# One-Hot Encoding is used to convert data into numeric form in order to fit it with Neural Network

automobile_features = pd.get_dummies(automobile_features, columns=['make', 'fuel-type', 'body-style'], dtype=float)

print(automobile_features.head())

print(automobile_features.columns)

# Let do some pre-processing before fitting the model to the NN using mean and std
automobile_features[['horsepower']] = preprocessing.scale(automobile_features[['horsepower']])

print(automobile_features[['horsepower']].head())

X_train, x_test, Y_train, y_test = train_test_split(automobile_features,
                                                    automobile_target,
                                                    test_size=0.2,
                                                    random_state=0)

dtype = torch.float
X_train_tensor = torch.tensor(X_train.values, dtype=dtype)
x_test_tensor = torch.tensor(x_test.values, dtype=dtype)

Y_train_tensor = torch.tensor(Y_train.values, dtype=dtype)
y_test_tensor = torch.tensor(y_test.values, dtype=dtype)

print(X_train_tensor.shape)
print(Y_train_tensor.shape)

# Constants for the Neural Network
inp = 26
out = 1
hid = 100  # Hidden Layer going to have 100 neurons
loss_fn = torch.nn.MSELoss()  # loss function is Mean Square Error/ Cost Function
learning_rate = 0.0001

# Fitting data for the Neural Network with Sigmoid Activation Fuction
# Its a fully connected NN because the all neurons are connect to it adjecent layer
model = torch.nn.Sequential(torch.nn.Linear(inp, hid),
                            torch.nn.Sigmoid(),
                            torch.nn.Linear(hid,out))

# Running Training for the 200000 epochs
# This is the Gradient Function Manually Designed
for iter in range(200000):
    y_pred = model(X_train_tensor)
    loss = loss_fn(y_pred, Y_train_tensor)
    if iter % 10000 == 0:
        print(iter, loss.item())

    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

# Let's take the sample data for testing and perform prediction
sample = x_test.iloc[23]
print(sample)

sample_tensor = torch.tensor(sample.values, dtype=dtype)
print(sample_tensor)

y_pred = model(sample_tensor)
print('Predicted price of automobile is: ', int(y_pred.item()))
print('Actual price of automobile is: ', int(y_test.iloc[23]))

# Plotting data
y_pred_tensor = model(x_test_tensor)
y_pred = y_pred_tensor.detach().numpy()

plt.scatter(y_pred, y_test.values)
plt.xlabel('Actual price')
plt.ylabel('Predicted price')
plt.title('Automobile Price Prediction')
plt.legend()
plt.show()

# Saving and loading our model
torch.save(model, 'My_AutoPrice_Model')
saved_model = torch.load('My_AutoPrice_Model')

# Plotting plot via saved model
y_pred_tensor = saved_model(x_test_tensor)
y_pred = y_pred_tensor.detach().numpy()

plt.figure(figsize=(15, 6))
plt.plot(y_pred, label='Predicted price')
plt.plot(y_test.values, label='Actual price')

plt.title('Automobile Price Prediction')
plt.legend()
plt.show()
