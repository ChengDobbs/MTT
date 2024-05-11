##################################
## task: t-SNE visualization of the MTT
## method: t-SNE
## input: cifar100_50/images_best.pt
## output: tsne.png
## cifar100-MTT
## class:100
## ipc: 50
##################################

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import pdb
from matplotlib.colors import ListedColormap

total_class_num = 100
selected_class_num = 5

def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

model_teacher = models.resnet18(num_classes=100)
model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model_teacher.maxpool = nn.Identity()
model_teacher = torch.nn.DataParallel(model_teacher).cuda()
checkpoint = torch.load('/root/SRE_repro/save/cifar100/resnet18_E200/ckpt.pth')
model_teacher.load_state_dict(checkpoint["state_dict"])
model_teacher = nn.Sequential(*(list(model_teacher.children())[:-1]))
model_teacher.eval()

tsne = TSNE(n_components=2, random_state=42)
# embedded_data_distill = tsne.fit_transform(features_distill_flat)
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

##################################

from torchvision.datasets import CIFAR100
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

cifar100_dataset = CIFAR100(root='/root/SRE_repro/data', train=True, download=True, transform=transform)
cifar100_loader = DataLoader(cifar100_dataset, batch_size=100, shuffle=False)
classes = np.random.choice(total_class_num, selected_class_num, replace=False)

flag = True
for ci in range(len(classes)):
    class_to_plot = classes[ci]
    print('class_to_plot:', class_to_plot)
    with torch.no_grad():
        for images, batch_labels in cifar100_loader:
            images = images.cuda()
            features_batch = model_teacher(images)
            labels_batch = batch_labels.cpu().numpy()
            # # 找到第一个类别的索引
            class_indices = np.where(labels_batch == class_to_plot)[0]
            print('class_indices:', class_indices)
            if len(class_indices) == 0:
                continue
            if flag:
                flag = False
                features_batch = features_batch[class_indices].squeeze().cpu().numpy()
                # print(features_batch.shape)
                features = features_batch
                print('c', ci, 'total', features.shape)
            else:
                features_batch = features_batch[class_indices].cpu().numpy()
                # print(features_batch.shape)
                features = np.concatenate((features, features_batch), axis=0)
                print('c', ci, 'total', features.shape)
    print(features.shape)

# features = np.vstack(features, axis=0)
features_flat = features.reshape(features.shape[0], -1)
features_flat = normalize_array(features_flat)
# # print(features_flat_concat.shape)
# # labels = np.concatenate(labels, axis=0)

# # 使用 t-SNE 将数据降维到二维
# embedded_data = tsne.fit_transform(features_flat)

# # 绘制 t-SNE 图
# for i in range(len(classes)):
#     plt.scatter(
#         embedded_data[classes[i] * 50:(classes[i] + 1) * 50, 0], 
#         embedded_data[classes[i] * 50:(classes[i] + 1) * 50, 1], 
#         label=str(classes[i])+'C100', 
#         marker='+'
#     )
# plt.legend()
# plt.savefig('tsne_c100.png')


##################################

# Load the data
mtt_pt = torch.load('../cifar100_50/images_best.pt')
# TypeError: pic should be PIL Image or ndarray. Got <class 'torch.Tensor'>

# for c in classes:
#     images_selected.append(images[c * 50:(c + 1) * 50])
# print(len(images_selected))

# calc 3 dim mean and std
# print(mtt_pt.shape)
mean_mtt = torch.mean(mtt_pt, dim=(0, 2, 3))
std_mtt = torch.std(mtt_pt, dim=(0, 2, 3))

# 将数据转换为 PyTorch 张量，并进行预处理
preprocess = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Resize((32, 32)),
    transforms.Normalize(mean=mean_mtt, std=std_mtt)
])

data = [preprocess(image) for image in mtt_pt]
data = torch.stack(data)
data = data.cuda()

with torch.no_grad():
    features_distill = model_teacher(data)

features_distill = features_distill.squeeze().cpu().numpy()
features_distill_flat = features_distill.reshape(features_distill.shape[0], -1)
features_distill_flat = normalize_array(features_distill_flat)

# [5000, 3, 32, 32] + [5000, 3, 32, 32]
features_flat_conbimed = np.concatenate((features_flat, features_distill_flat), axis=0)

# embedded_data_distill = tsne.fit_transform(features_distill_flat)
embedded_data_conbimed = tsne.fit_transform(features_flat_conbimed)

# for i in range(10):
    # plt.scatter(embedded_data_conbimed[classes[i] * 50:(classes[i] + 1) * 50, 0], embedded_data_conbimed[classes[i] * 50:(classes[i] + 1) * 50, 1], label=str(classes[i])+' MTT', marker='+', c=colors[i])

for i in range(selected_class_num):
    plt.scatter(embedded_data_conbimed[classes[i] * 50:(classes[i] + 1) * 50, 0], embedded_data_conbimed[classes[i] * 50:(classes[i] + 1) * 50, 1], label=str(classes[i])+' C100', marker='+', c=colors[i])
    plt.scatter(embedded_data_conbimed[classes[i] * 50 + 5000:(classes[i] + 1) * 50 + 5000, 0], embedded_data_conbimed[classes[i] * 50 + 5000:(classes[i] + 1) * 50 + 5000, 1], label=str(classes[i])+' MTT', marker='*', c=colors[i])

plt.legend()
# plt.savefig('MTT cifar100_50')
plt.savefig('tsne_MTT_mix.png')
# plt.show()