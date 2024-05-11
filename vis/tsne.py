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
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pdb

from matplotlib.colors import ListedColormap

classes = np.random.choice(100, 10, replace=False)


# Load the data
images = torch.load('cifar100_50/images_best.pt')

print(images)

# std = torch.std(images)
# mean = torch.mean(images)
# images = (images - mean) / std

print(classes)
images_selected = []

for c in classes:
    images_selected.append(images[c * 50:(c + 1) * 50])
print(len(images_selected))
# images_selected = torch.cat(images_selected, dim=0)
# labels_selected = torch.cat(labels_selected, dim=0)

# Flatten the images
# images_selected = images_selected.view(images_selected.size(0), -1)

# Perform t-SNE
# tsne = TSNE(n_components=2, random_state=0)
# image_mtt = tsne.fit_transform(images_selected)

# Plot the t-SNE
# make the colors more separated
# plt.figure(figsize=(10, 10))

# for i in range(10):
#     plt.scatter(image_mtt[i * 50:(i + 1) * 50, 0], image_mtt[i * 50:(i + 1) * 50, 1], label=str(classes[i]), marker='*')
# titled as MTT
# no x-axis and y-axis
# legend
# plt.title('MTT')

# plt.savefig('tsne-MTT.png')


##################################
## task: t-SNE visualization of the SRe2L
## method: t-SNE
## input: /root/SRE_repro/syn_data/cifar100_rn18_1k_ipc50/sre_best.pt
## output: tsne.png
## cifar100 - SRe2L
## class: 100
## ipc: 50
##################################

# Load the data
images = torch.load('/root/SRE_repro/syn_data/cifar100_rn18_1k_ipc50/sre_best.pt')

print(classes)
# images_selected = []
std = torch.std(images)
mean = torch.mean(images)
images = (images - mean) / std

for c in classes:
    images_selected.append(images[c * 50:(c + 1) * 50])
print(len(images_selected))

images_selected = torch.cat(images_selected, dim=0)
print(images_selected.shape)

# Flatten the images
images_selected = images_selected.view(images_selected.size(0), -1)

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=0)
images_tsne = tsne.fit_transform(images_selected)

# Plot the t-SNE
# make the colors more separated
plt.figure(figsize=(10, 10))

print(images_tsne.shape)

for i in range(10):
    plt.scatter(images_tsne[i * 50:(i + 1) * 50, 0], images_tsne[i * 50:(i + 1) * 50, 1], label=str(classes[i])+'MTT', marker='^')

for i in range(10,20):
    plt.scatter(images_tsne[i * 50:(i + 1) * 50, 0], images_tsne[i * 50:(i + 1) * 50, 1], label=str(classes[i-10])+'SRe2L', marker='*')
# titled as SRe2L
# no x-axis and y-axis
# legend
plt.title('SRe2L&MTT')
plt.xticks([])
plt.yticks([])
plt.legend(loc=10, ncol=2)
plt.savefig('tsne-mix.png')

##################################
## task: t-SNE visualization of the SRe2L-in1k
## method: t-SNE
## input: /root/SRE_repro/syn_data/sre2l_in1k_rn18_2k_ipc50
## output: tsne-in1k.png
## cifar100 - SRe2L
## class: 1000
## ipc: 50
##################################

def SRe2L_in1k():
    from PIL import Image

    # load from syn_data/sre2l_in1k_rn18_2k_ipc50
    # new000/class000_id000.jpg
    # ...
    # new999/class999_id049.jpg
    # 10 classes selected, 50 images per class selected
    # cls0: images.data[0:50]
    # cls1: images.data[50:100]
    # ...
    # cls99: images.data[4950:5000]

    classes = np.random.choice(1000, 10, replace=False)
    print(classes)
    images_selected = []

    for c in classes:
        images = torch.tensor([])
        for i in range(50):
            img = Image.open(f'/root/SRE_repro/syn_data/sre2l_in1k_rn18_2k_ipc50/new{c:03d}/class{c:03d}_id{i:03d}.jpg')
            img = transforms.ToTensor()(img)
            images = torch.cat((images, img.unsqueeze(0)), dim=0)
        images_selected.append(images)

    # flatten the images
    images_selected = torch.cat(images_selected, dim=0)
    # print(images_selected.shape)

    # # select labels: 50 consecutive c[0], ..., 50 consecutive c[9]
    labels_selected = [c for c in classes for _ in range(50)]
    # print(labels_selected)

    # Flatten the images
    images_selected = images_selected.view(images_selected.size(0), -1)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    images_tsne = tsne.fit_transform(images_selected)

    # Plot the t-SNE
    # make the colors more separated
    # plt.figure(figsize=(10, 10))

    for i in range(10):
        plt.scatter(images_tsne[i * 50:(i + 1) * 50, 0], images_tsne[i * 50:(i + 1) * 50, 1], label=str(classes[i]), marker='+')
    # titled as SRe2L
    # no x-axis and y-axis
    # legend
    # plt.title('SRe2L sre2l_in1k_rn18_2k_ipc50')
    # plt.xticks([])
    # plt.yticks([])
    plt.legend()
    # plt.savefig('tsne-mix.png')
    # plt.show()

# if __name__ == '__main__':
    # MTTtSNE()
    # print('MTT Done.')
    # SRe2LtSNE()
    # print('SRe2L Done.')
    SRe2L_in1k()
    print('SRe2L_in1k Done.')