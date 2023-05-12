import numpy as np
import matplotlib.pyplot as plt
import torch

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]],
                   dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]],
                   dtype=np.float32)

plt.plot(x_train, y_train, 'ro', label='Original Data')
plt.show()

# Input and Output features have requires_grad False because these features are not for training
X_train = torch.from_numpy(x_train)
Y_train = torch.from_numpy(y_train)
print('requires_grad for X_train: ', X_train.requires_grad)
print('requires_grad for Y_train: ', Y_train.requires_grad)

# Constants for the NN with the 1 layer and 100 neurons and ReLU activation
input_size = 1
hidden_size = 100
output_size = 1
learning_rate = 1e-6
print(learning_rate)

w1 = torch.rand(input_size, hidden_size, requires_grad=True)
w1.shape

w2 = torch.rand(hidden_size, output_size, requires_grad=True)
w2.shape

# 300 Epoch's iteration mm(w1) & mm(w2) is the Affine Transformation Function & clamp(min=0) for clamp negative
# numbers to 0 i.e. ReLU Activation
for iter in range(1, 301):
    y_pred = X_train.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - Y_train).pow(2).sum()  # Mean Square Error(MSE)

    if iter % 50 == 0:
        print(iter, loss.item())

    loss.backward()
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()

print('w1: ', w1)
print('w2: ', w2)

x_train_tensor = torch.from_numpy(x_train)
y_predicted_in_tensor = x_train_tensor.mm(w1).clamp(min=0).mm(w2)

predicted = y_predicted_in_tensor.detach().numpy()
print(predicted)

plt.plot(x_train, y_train, 'ro', label='Original Data')
plt.plot(x_train, predicted, label='Fitted Line')
plt.legend()
plt.show()