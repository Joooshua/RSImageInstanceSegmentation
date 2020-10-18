import os
import numpy as np
import random
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import sync_transforms


matches = [100, 200, 300, 400, 500, 600, 700, 800]
images_path = './train/images/'
labels_path = './train/labels/'
img_name_list = os.listdir(images_path)
label_name_list = os.listdir(labels_path)
training_samples = int(len(img_name_list) * 0.99)

def get_img_label_paths(images_path, labels_path):
    res = []
    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)):
            file_name, _ = os.path.splitext(dir_entry)
            res.append((os.path.join(images_path, file_name+".tif"),
                        os.path.join(labels_path, file_name+".png")))
    return res

def rotate_bound(img, angle):
    if angle == 90:
        out = Image.fromarray(img).transpose(Image.ROTATE_90)
        return np.array(out)
    if angle == 180:
        out = Image.fromarray(img).transpose(Image.ROTATE_180)
        return np.array(out)
    if angle == 270:
        out = Image.fromarray(img).transpose(Image.ROTATE_270)
        return np.array(out)

def randomColor(image):

    random_factor = random.choice([-30,0,30,60,90])
    brightness = np.ones(image.shape, dtype = 'uint16') * random_factor
    return cv2.subtract(image, brightness)

def data_augment(x, y):
    flag = random.choice([1,2,3,4,5,6])
    if flag == 1:
        x, y = cv2.flip(x, 1), cv2.flip(y, 1)  # Horizontal mirror
    if flag == 2:
        x, y = cv2.flip(x, 0), cv2.flip(y, 0)  # Vertical mirror
    if flag == 3:
        x, y = rotate_bound(x, 90), rotate_bound(y, 90)
    if flag == 4:
        x, y = rotate_bound(x, 180), rotate_bound(y, 180)
    if flag == 5:
        x, y = rotate_bound(x, 270), rotate_bound(y, 270)
    else:
        pass
    return x, y

#resize_scale_range = [float(scale) for scale in '0.5, 2.0'.split(',')]
img_transform = transforms.Compose([
    #transforms.RandomRotation(degrees=15),
    #transforms.ColorJitter(),
    #transforms.RandomHorizontalFlip(),
    #transforms.CenterCrop(size=256),
    #sync_transforms.RandomScale(256, 256, resize_scale_range),
    #sync_transforms.RandomFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class MaskToTensor(object):
    def __call__(self, mask):
        return torch.from_numpy(np.array(mask, dtype=np.int32)).long()

label_transform = MaskToTensor()


class RSDataset(Dataset):
    def __init__(self, img_label_pairs, img_transform, label_transform, train=True):
        train_img_label_pairs = img_label_pairs[:training_samples]
        val_img_label_pairs = img_label_pairs[training_samples:]

        if train:
            self.img_label_path = train_img_label_pairs
        else:
            self.img_label_path = val_img_label_pairs

        self.img_transform = img_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        img = cv2.imread(self.img_label_path[index][0], cv2.IMREAD_UNCHANGED)
        label = cv2.imread(self.img_label_path[index][1], cv2.IMREAD_UNCHANGED)

        for m in matches:
            label[label == m] = matches.index(m)
        '''
        # seg_labels = np.zeros((256, 256, nClasses))
        for c in range(nClasses):
            seg_labels[:, :, c] = (label == c).astype(int)
        '''
        img, label = data_augment(img, label)
        return img_transform(img), self.label_transform(label)

    def __len__(self):
        return len(self.img_label_path)

img_label_pairs = get_img_label_paths(images_path, labels_path)
random.shuffle(img_label_pairs)

train_dataset = RSDataset(img_label_pairs, img_transform, label_transform, train=True)
val_dataset = RSDataset(img_label_pairs, img_transform, label_transform, train=False)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)
