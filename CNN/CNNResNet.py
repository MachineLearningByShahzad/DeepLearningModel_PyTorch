import torch
from torchvision import datasets, models, transforms
import zipfile

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.Normalize(mean, std)
                                ])

test_transform = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                                ])

# zip = zipfile.ZipFile('../CNN/datasets/flowers.zip')
# zip.extractall('datasets')

data_dir = './CNN/datasets/FlowersDatasets'
image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(data_dir + '/train', train_transform)
image_datasets['test'] = datasets.ImageFolder(data_dir + '/test', train_transform)

print('Training data size - %d' % len(image_datasets['train']))
print('Testing data size - %d' % len(image_datasets['test']))

class_names = image_datasets['train'].classes
print(class_names)

image_datasets
