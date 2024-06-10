from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
# from tqdm.notebook import tqdm
from tqdm import tqdm
import cv2, os, glob
import rasterio 
from efficient import ViT
import gc

batch_size = 16
epochs = 500
lr = 3e-5
gamma = 0.7
seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

device = 'cuda'

train_dir = '/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/all_data_training/DATA_TRAIN_VIT/train'
test_dir = '/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/all_data_training/DATA_TRAIN_VIT/test'

train_list = glob.glob(os.path.join(train_dir,'*.tif'))
test_list = glob.glob(os.path.join(test_dir, '*.tif'))

print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")

labels = [path.split('/')[-1].split('_')[0] for path in train_list]
np.unique(labels)

train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=seed)

print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        with rasterio.open(img_path,'r') as src:
            img = src.read()
            img = img.transpose(1,2,0)
            # img = np.array(img)
            img1 = img[:,:,0:4]
            img1 = Image.fromarray(img1)
            img2 = img[:,:,4:8]
            img2 = Image.fromarray(img2)
    
        img_transformed1 = self.transform(img1)
        img_transformed2 = self.transform(img2)
        img_transformed = torch.cat((img_transformed1, img_transformed2), dim=0)
        del img1, img2, img_transformed1, img_transformed2
        label = img_path.split("/")[-1].split("_")[0]
        label = 1 if label == "change" else 0
        # print(label)
        # print(img_transformed.shape)
        return img_transformed, label

train_data = CatsDogsDataset(train_list, transform=train_transforms)
valid_data = CatsDogsDataset(valid_list, transform=test_transforms)
test_data = CatsDogsDataset(test_list, transform=test_transforms)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8 )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

efficient_transformer = Linformer(
    dim=128,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)


model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=8,
).to(device)

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


weight_dir = '/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/weight_changedetection/classify'

checkpoint = torch.load('/home/skymap/big_data/Giang_workspace/Gitlab/change_detection_storage/weight_changedetection/classify/classify_38.pth')
model.load_state_dict(checkpoint)

best_val_loss = 100
for epoch in range(38,epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    #region train
    print('train start')
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        # print('aaaaaa',len(data))
        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        
        del data, label, output, loss, acc
        gc.collect()
        torch.cuda.empty_cache()
    #endregion
    
    #region validation
    print('validation start')
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in tqdm(valid_loader):
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

        if best_val_loss >= val_loss / len(valid_loader):
            torch.save(model.state_dict(), os.path.join(weight_dir,f'classify_{epoch}.pth'))   
            best_val_loss = val_loss/len(valid_loader) 
            print(f"Save new best model at epoch {epoch}")
            
    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
    #endregion