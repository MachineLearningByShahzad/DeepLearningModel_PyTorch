import pandas as pd
import random
from sklearn import preprocessing
import string
import torch

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

genders=['Female', 'Male']
all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)
print(all_letters)

# Converting Name to Tensors
def name_to_tensor(name):
    name_in_tensor = torch.zeros(len(name), 1, n_letters)

