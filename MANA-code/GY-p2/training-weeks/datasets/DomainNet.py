from os import path
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


def read_domainnet_data(dataset_path, domain_name, split="train"):
    data_paths = []
    data_labels = []
    split_file = path.join(dataset_path, "splits", "{}_{}.txt".format(domain_name, split))
    with open(split_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            data_path, label = line.split(' ')
            data_path = path.join(dataset_path, data_path)
            label = int(label)
            data_paths.append(data_path)
            data_labels.append(label)
    return data_paths, data_labels


class DomainNet(Dataset):
    def __init__(self, data_paths, data_labels, transforms, domain_name):
        super(DomainNet, self).__init__()
        self.data_paths = data_paths
        self.data_labels = data_labels
        self.transforms = transforms
        self.domain_name = domain_name

    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not img.mode == "RGB":
            img = img.convert("RGB")
        label = self.data_labels[index]
        img = self.transforms(img)

        return img, label

    def __len__(self):
        return len(self.data_paths)


def get_domainnet_dloader(base_path, domain_name, batch_size, num_workers):
    dataset_path = base_path  # path.join(base_path, 'dataset', 'DomainNet')
    train_data_paths, train_data_labels = read_domainnet_data(dataset_path, domain_name, split="train")
    test_data_paths, test_data_labels = read_domainnet_data(dataset_path, domain_name, split="test")
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    transforms_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    # transforms_train = transforms.Compose([
    #     transforms.RandomResizedCrop(96, scale=(0.75, 1)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor()
    # ])
    # transforms_test = transforms.Compose([
    #     transforms.Resize((96,96)),
    #     transforms.ToTensor()
    # ])
    # transforms_train = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.ToTensor()
    # ])
    # transforms_test = transforms.Compose([
    #     transforms.ToTensor()
    # ])

    train_dataset = DomainNet(train_data_paths, train_data_labels, transforms_train, domain_name)
    test_dataset = DomainNet(test_data_paths, test_data_labels, transforms_test, domain_name)
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                               shuffle=True)
    test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                              shuffle=True)
    return train_dloader, test_dloader


def get_domainnet_dloader_train(base_path, domain_data, domain_label, batch_size, num_workers):
    
    Transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    length = 4
    indices = np.random.choice(len(domain_label), size=length, replace=False)
    test_data = np.array(domain_data)[indices]
    test_label = np.array(domain_label)[indices]
    domain_data = np.delete(domain_data, indices, axis=0)
    domain_label = np.delete(domain_label, indices)
    
    train_dataset = CustomDataset(domain_data, domain_label, transform=Transform)
    test_dataset = CustomDataset(test_data, test_label, transform=Transform)
    
    train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    test_dloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    
    return train_dloader,test_dloader


def get_domainnet_dloader_test(base_path, domain_data, domain_label, batch_size, num_workers, mix_length):

    Transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # length = 30
    length = mix_length # for single-day testing
    indices = np.random.choice(len(domain_label), size=length, replace=False)
    train_data = np.array(domain_data)[indices]
    train_label = np.array(domain_label)[indices]
    domain_data = np.delete(domain_data, indices, axis=0)
    domain_label = np.delete(domain_label, indices)

    train_dataset = CustomDataset(train_data, train_label, transform=Transform)
    test_dataset = CustomDataset(domain_data, domain_label, transform=Transform)

    train_dloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    test_dloader = DataLoader(test_dataset, batch_size=1, num_workers=num_workers, pin_memory=True, shuffle=True)

    return  train_dloader,test_dloader

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            if isinstance(sample, torch.Tensor):
                sample = sample.numpy()
            sample = self.transform(sample)
        return sample, label
    

def load_train_data_8_class_multi_domain_train(domain_name):
    # all self new
    train_path = '/home/wangqingyu/IL/dataset/DomainNet/train_8_class_choose/'
    Flag = 0
    # import pdb; pdb.set_trace()

    if isinstance(domain_name, list):
    
        for i in range(42):  # 42 domain
        
            if i in domain_name:
                file_data = train_path + str(i) + '.npz'
                train_data = np.load(file_data, allow_pickle=True)
                x_train = train_data['data'][()]['x']  # (36096, 3, 64, 64)
                y_train = train_data['data'][()]['y']

                del train_data

                if Flag == 0:
                    x_train_self = x_train
                    y_train_self = y_train
                    Flag = 1
                else:
                    x_train_self = np.concatenate((x_train_self, x_train), axis=0)
                    y_train_self = np.concatenate((y_train_self, y_train), axis=0)
            else:
                continue
            
    else:
        
        file_data = train_path + str(domain_name) + '.npz'
        train_data = np.load(file_data, allow_pickle=True)
        x_train = train_data['data'][()]['x']  # (36096, 3, 64, 64)
        y_train = train_data['data'][()]['y']

        del train_data

        x_train_self = x_train
        y_train_self = y_train
        
    return x_train_self, y_train_self


def load_test_data_8_class_multi_domain_test(domain_name):
    # 6
    # all self new
    test_path = '/home/wangqingyu/IL/dataset/DomainNet/test_8_class_choose/'
    Flag = 0
    if isinstance(domain_name, list):
        
        for i in range(42):  # 42 domain
        
            if i in domain_name:
                file_data = test_path + str(i) + '.npz'
                train_data = np.load(file_data, allow_pickle=True)
                x_train = train_data['data'][()]['x']  # (36096, 3, 64, 64)
                y_train = train_data['data'][()]['y']

                del train_data

                if Flag == 0:
                    x_test_new = x_train
                    y_test_new = y_train
                    Flag = 1
                else:
                    x_test_new = np.concatenate((x_test_new, x_train), axis=0)
                    y_test_new = np.concatenate((y_test_new, y_train), axis=0)
            else:
                continue
                    
    else:
        
        file_data = test_path + str(domain_name) + '.npz'
        test_data = np.load(file_data, allow_pickle=True)
        x_test = test_data['data'][()]['x']  # (36096, 3, 64, 64)
        y_test = test_data['data'][()]['y']

        del test_data

        x_test_new = x_test
        y_test_new = y_test
            
    return x_test_new, y_test_new



