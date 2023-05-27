import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn import preprocessing
import string
import torch
import torch.nn as nn
# from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix
import numpy as np

maleDF = pd.read_csv('../RNN/datasets/male.txt', header=None)
femaleDF = pd.read_csv('../RNN/datasets/female.txt', header=None)
maleDF.columns = ['Name']
femaleDF.columns = ['Name']
maleDF['Gender'] = 'Male'
femaleDF['Gender'] = 'Female'

names_data = pd.concat([maleDF, femaleDF], axis=0, ignore_index=True)
print(names_data.head())

print(len(names_data['Name']))
print(len(names_data['Name'].unique()))

names_data = names_data.drop_duplicates(subset='Name',
                                        keep=random.choice(['first', 'last']))

print(len(names_data))

# one hot encoding of the data
le = preprocessing.LabelEncoder()
names_data['Gender'] = le.fit_transform(names_data['Gender'])
print(names_data.head())

genders = ['Female', 'Male']
all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)
print(all_letters)


# Converting Name to Tensors
def name_to_tensor(name):
    name_in_tensor = torch.zeros(len(name), 1, n_letters)
    for i, letter in enumerate(name):
        name_in_tensor[i][0][all_letters.find(letter)] = 1

    return name_in_tensor


print(name_to_tensor('a'))


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128
n_gender = len(genders)

rnn = RNN(n_letters, n_hidden, output_size=n_gender)

iterations = 100000
criterion = nn.NLLLoss()
learning_rate = 0.005


def output_to_gender(output):
    top_n, top_index = output.topk(1)
    pred_i = top_index[0].item()
    pred = genders[pred_i]

    return pred


# Training a RNN
for iteration in range(1, iterations + 1):
    i = random.randint(0, len(names_data) - 1)

    name = names_data.iloc[i][0]
    name_in_tensor = name_to_tensor(name)

    gender = names_data.iloc[i][1]
    gender_in_tensor = torch.LongTensor([gender])

    hidden = rnn.initHidden()
    rnn.zero_grad()

    for i in range(name_in_tensor.size()[0]):
        output, hidden = rnn(name_in_tensor[i], hidden)

    loss = criterion(output, gender_in_tensor)
    loss.backward()

    for p in rnn.parameters():
        # p.data.add_(-learning_rate, p.grad.data)
        p.data.add_(p.grad.data, alpha=-learning_rate)
    if iteration % 5000 == 0:
        pred = output_to_gender(output)
        correct = '✓' if pred == genders[gender] else '✗ (%s)' % genders[gender]

        print('iters - %d %d%% (%s) Name- %s Gender- %s %s' % \
              (iteration, iteration/iterations * 100, loss.item(), name, pred, correct))

# Creating a ConfusionMatrix
n_confusion = 10000
prediction = []
actual = []

for _ in range(n_confusion):
    i = random.randint(0, len(names_data) - 1)

    name = names_data.iloc[i][0]
    name_in_tensor = name_to_tensor(name)

    gender_idx = names_data.iloc[i][1]
    gender_in_tensor = torch.LongTensor([gender_idx])

    hidden = rnn.initHidden()

    for j in range(name_in_tensor.size()[0]):
        output, hidden = rnn(name_in_tensor[j], hidden)

    pred = output_to_gender(output)

    prediction.append(pred)
    actual.append(genders[gender_idx])

np_prediction = np.array(prediction)
np_actual = np.array(actual)

# cm = ConfusionMatrix(np.where(np_prediction == 'Female', True, False),
#                      np.where(np_actual == 'Female', True, False))

# confusion_matrix(true_labels, predicted_labels)
cm = confusion_matrix(np.where(np_prediction == 'Female', True, False),
                     np.where(np_actual == 'Female', True, False))

print(cm)

# Create a heatmap
plt.imshow(cm, cmap='Blues')

# Add color-bar
plt.colorbar()

# Add labels
classes = ['Male', 'Female']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Add text annotations
thresh = cm.max() / 2.0
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center',
             color='white' if cm[i, j] > thresh else 'black')

# Set axis labels
plt.xlabel('Predicted label')
plt.ylabel('True label')

# Show the plot
plt.show()
