import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torchvision
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from skimage.io import imread

from imagenet_ipc import ImageFolderIPC

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = imread(os.path.join(folder, filename))
        images.append(img)
    return images

def normalize_array(array):
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array

# 加载 in1k 数据集

train_dir = '/root/SRE_repro/syn_data/in1k_rn18_awp_2k_ipc50'
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])

in1k_dataset = ImageFolderIPC(
    train_dir,
    transform=train_transforms,
    image_number=50
)

in1k_loader = torch.utils.data.DataLoader(
    in1k_dataset, 
    batch_size=64, shuffle=True,
    num_workers=16, pin_memory=True)

# 加载预训练的 ResNet-18 模型
model_teacher = torchvision.models.resnet18(pretrained = True)
model_teacher.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model_teacher.maxpool = nn.Identity()
# move the FC layer to the end
# model_teacher = nn.Sequential(*list(model_teacher.module.children())[:-2])
model_teacher = nn.Sequential(*(list(model_teacher.children())[:-2]))

model_teacher = torch.nn.DataParallel(model_teacher).cuda()
# checkpoint = torch.load('/data/zhangxin/Ortho_SRE-main/save/cifar100/resnet18_E200/ckpt.pth')
# model_teacher.load_state_dict(checkpoint["state_dict"])
model_teacher.eval()

tsne = TSNE(n_components=2, random_state=42)

classes = np.random.choice(1000, 10, replace=False)
print(classes)
plt.figure(figsize=(10, 8))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

features = []
flag = True
for ci in range(len(classes)):
    class_to_plot = classes[ci]
    
    with torch.no_grad():
        for _, (images, batch_labels) in enumerate(in1k_loader):
            images= images.cuda()
            features_batch = model_teacher(images)
            labels_batch = batch_labels.cpu().numpy()
            # # 找到第一个类别的索引
            class_indices = np.where(labels_batch == class_to_plot)[0]
            if flag:
                if len(class_indices) > 0:  # 如果当前批次包含第一个类的样本
                    flag = False
                    features_batch = features_batch[class_indices].squeeze().cpu().numpy()
                    features = features_batch.reshape(-1, features_batch.shape[0], features_batch.shape[1], features_batch.shape[2])
                    print('c', ci, 'total', features.shape[0])
            else:
                if len(class_indices) > 0:  # 如果当前批次包含第一个类的样本
                    # wrong
                    features_batch = features_batch[class_indices].cpu().numpy()
                    features = np.concatenate((features, features_batch), axis=0)
                    print('c', ci, 'total', features.shape[0])
    print(features.shape)

    # features = np.vstack(features, axis=0)
features_flat = features.reshape(features.shape[0], -1)
features_flat = normalize_array(features_flat)
    # print(features_flat_concat.shape)
    # labels = np.concatenate(labels, axis=0)

# 使用 t-SNE 将数据降维到二维
embedded_data = tsne.fit_transform(features_flat)

# 绘制 t-SNE 图
for i in range(len(classes)):
    plt.scatter(embedded_data[i * 50:(i + 1) * 50, 0], embedded_data[i * 50:(i + 1) * 50, 1], label=str(classes[i]), marker='+')
plt.legend()
plt.savefig('tsne_in1k.png')


# # 将数据转换为 PyTorch 张量，并进行预处理
# preprocess = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize((224, 224)),  # 调整大小到模型预期的大小
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                      std=[0.229, 0.224, 0.225])  # 归一化
# ])

# data = [preprocess(image) for image in data]
# data = torch.stack(data)
# data = data.cuda()

# with torch.no_grad():
#     features_distill = model_teacher(data)

# features_distill = features_distill.squeeze().cpu().numpy()
# features_distill_flat = features_distill.reshape(features_distill.shape[0],-1)
# features_distill_flat = normalize_array(features_distill_flat)
# tsne = TSNE(n_components=2, random_state=42)
# embedded_data_distill = tsne.fit_transform(features_distill_flat)
# colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
# for i in range(10):
#     plt.scatter(embedded_data_distill[i * 50:(i + 1) * 50, 0], embedded_data_distill[i * 50:(i + 1) * 50, 1], label=str(classes[i]), marker='+', c=colors[i])

# plt.title('t-SNE Visualization of in1k Data')
# plt.legend()
# plt.savefig('tsne_in1k.png')
# plt.show()
