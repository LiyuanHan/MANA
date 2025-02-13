import numpy as np
import torch
import os

'''
:data_size   mnist 28   dvs 128
:label_len   mnist 1000 dvs 1464

'''

data_size = 28
label_len = 1000


all = []
all_images = np.empty((0, data_size), dtype=np.float32)    # Store stacked data and labels
all_labels = np.empty((0, 1), dtype=np.int64)
labels_list = np.empty((0, label_len), dtype=np.int64)
len_of_each_session = []

# Specify the save path
save_path = "save_path"
# Check if the directory exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

for i in range(12):
    file_path = f'./npz_path/name.npz'
    if os.path.exists(file_path):
        data = np.load(file_path, allow_pickle=True)
        # Perform the required operations
        print("File has been loaded")
    else:
        print("File does not exist, skipping")
        continue

    train_dict = data['data'].item()

    # Get image data and label data
    image_data = train_dict['x']
    image_data = np.transpose(image_data, (0, 1, 3, 2))
    labels = train_dict['y']
    labels_y = labels.reshape(-1, 1)

    images_concat = image_data.reshape(-1, image_data.shape[2])
    labels_concat = np.repeat(labels_y, image_data.shape[2], axis=0)

    all_images = np.vstack((all_images, images_concat))
    all_labels = np.vstack((all_labels, labels_concat))

    labels_target = labels.reshape(-1, label_len)
    labels_list = np.vstack((labels_list, labels_target))

    len_of_each_session.append([data_size] * label_len)

all_images = torch.tensor(all_images)
all_labels = torch.tensor(all_labels)
print(all_images.shape, all_labels.shape)
all = labels_list.tolist()
print(len(all))
print(len(len_of_each_session))

data_path = save_path + '/x.pt'
labels_path = save_path + '/y.pt'
target_path = save_path + '/target.pt'
len_of_each_session_path = save_path + '/len_of_each_session.pt'

torch.save(all_images, data_path)
torch.save(all_labels, labels_path)
torch.save(all, target_path)
torch.save(len_of_each_session, len_of_each_session_path)