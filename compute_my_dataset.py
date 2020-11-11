from torchvision import transforms, datasets, models
import numpy as np
from PIL import Image
from my_dataset import NightDataSet
import os

# 计算自己数据集的均值方差
def compute_mean_and_std(dataset):
    # 均值计算
    mean_r = 0
    mean_g = 0
    mean_b = 0
    for img_id in dataset.ids:
        print(img_id)
        img_path = os.path.join(dataset.img_root, 'sample' + img_id + '.jpg')
        img = Image.open(img_path)
        img = np.array(img)
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(dataset)
    mean_g /= len(dataset)
    mean_r /= len(dataset)

    # 方差计算
    diff_r = 0
    diff_g = 0
    diff_b = 0

    N = 0
    for img_id in dataset.ids:
        print(img_id)
        img_path = os.path.join(dataset.img_root, 'sample' + img_id + '.jpg')
        img = Image.open(img_path)
        img = np.array(img)
        diff_b = np.sum(np.power(img[:, :, 0] - mean_b, 2))
        diff_g = np.sum(np.power(img[:, :, 1] - mean_g, 2))
        diff_r = np.sum(np.power(img[:, :, 2] - mean_r, 2))
        N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / N)
    std_g = np.sqrt(diff_g / N)
    std_r = np.sqrt(diff_r / N)

    mean = (mean_b / 255.0, mean_g / 255.0, mean_r / 255.0)
    std = (std_b / 255.0, std_g / 255.0, std_r / 255.0)

    return mean, std


# 数据集预处理
night_root = "./"  # light data set path
all_dataset = NightDataSet(night_root, transforms.ToTensor(), train_set='all.txt')
mean, std = compute_mean_and_std(all_dataset)

print(mean)
print(std)
