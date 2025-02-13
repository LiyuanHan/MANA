import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

MNIST_path = 'MNIST_PATH'

def generate_data_train(MNIST_path, index, save_path, npz_name, method):
    '''
    :param MNIST_path: Path to the original dataset
    :param index: Time index for simulating neural data
    :param save_path: Path to save the generated data
    :param npz_name: Name of the simulated data
    :param method: Simulation method: "translate", "rotate", "mix"
    :return:
    '''
    # Number of samples per domain
    batch_size = 1000
    trans_perm = transforms.Compose([transforms.ToTensor()])
    dataset_train = datasets.MNIST(root=MNIST_path, train=True, transform=trans_perm, download=True)
    train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)

    dict_0_4 = {i: 0 for i in range(5)}
    dict_5_9 = {i: 2 for i in range(5, 10)}
    dict_10_14 = {i: 4 for i in range(10, 15)}
    dict_15_19 = {i: 6 for i in range(15, 20)}
    dict_20_24 = {i: 8 for i in range(20, 25)}
    dict_25_29 = {i: 10 for i in range(25, 30)}
    dict_30_34 = {i: 12 for i in range(30, 35)}

    combined_dict = {**dict_0_4, **dict_5_9, **dict_10_14, **dict_15_19, **dict_20_24, **dict_25_29, **dict_30_34}

    for i, tt in enumerate(train_loader):
        if i == index:
            train_data, train_label = tt
            dataset_image = []
            dataset_label = []

            for img, lab in zip(train_data, train_label):
                if method == "translate":
                    t_data = translate_image(img, index, method)
                elif method == "rotate":
                    t_data = Rotate(img, index, step=1, change_num=35, method=method)
                elif method == "mix":
                    t_data = translate_image(img, combined_dict[index],method=method)
                    t_data = Rotate(t_data, index, step=1, change_num=35, method=method)

                t_data = torch.unsqueeze(t_data, dim=0)
                dataset_image.extend(t_data.cpu().detach().numpy())
            dataset_label.extend(train_label.cpu().detach().numpy())

            image = np.array(dataset_image)
            label = np.array(dataset_label)

            train_list = []
            num_samples = {'train': []}
            train_list.append({'x': image, 'y': label})
            num_samples['train'].append(len(label))

            print("The number of train samples: ", num_samples['train'], index)

            for j, train_dict in enumerate(train_list):
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                with open(os.path.join(save_path, npz_name + str(index + 1) + '.npz'), 'wb') as f:
                    np.savez_compressed(f, data=train_dict)
            break
        else:
            continue


def translate_image(img, indx, method):
    if method == 'translate':
        x_translation = [0, 0, 0, 0, 0, 2, 4, 6, 8, 10, 12, 14]
    if method == 'mix':
        x_translation = [i for i in range(0, 14, 2) for _ in range(5)]

    if isinstance(img, torch.Tensor):
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))
    new_image = np.roll(img, x_translation[indx], axis=1)  # Translate horizontally
    if isinstance(new_image, np.ndarray):
        new_image = np.transpose(new_image, (2, 0, 1))
        new_image = torch.from_numpy(new_image)
    return new_image


def Rotate(img, i, step, change_num, method):
    if method == "rotate":
        degree_list = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
    if method == "mix":
        degree_list = [step * j for j in range(change_num)]

    degrees = degree_list
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    angle = degrees[i]
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(img, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    rotated_image = np.expand_dims(rotated_image, axis=-1)
    if isinstance(rotated_image, np.ndarray):
        rotated_image = np.transpose(rotated_image, (2, 0, 1))
        rotated_image = torch.from_numpy(rotated_image)
    return rotated_image


if __name__ == '__main__':
    save_path = 'save_path'
    npz_name = 'source_name'
    """
    translate rotate   num=12
    mix  num=35
    """
    num = 12
    for i in range(num):
        generate_data_train(MNIST_path, i, save_path, npz_name, method="method")