
import torch
from basic_model import CDEvaluator
import torchvision.transforms.functional as TF
import numpy as np
import rasterio
from rasterio.windows import Window
import threading
from tqdm import tqdm
import concurrent.futures
from config import Config
import torchvision.transforms as transforms
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from models.efficient import ViT
from PIL import Image


def predict_cf(path_img1, path_img2,cfg, out_path,flag):

        if flag=='sgd':
            cfg.n_class = 3
        if flag=='adam':
            cfg.n_class = 4
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device("cuda:0")
        model = CDEvaluator(cfg)
        model.load_checkpoint()
        model.net_G.to(device)
        print(next(model.net_G.parameters()).device)
        # exit()
        # from torchinfo import summary
        # model_summary = summary(model.net_G, input_size=([(224, 224, 4),(224, 224, 4)] ), col_names=["input_size", "output_size", "num_params"], verbose=0, depth=10)
        model.eval()
        num_bands = cfg.num_bands
        efficient_transformer = Linformer(
            dim=128,
            seq_len=49+1,  # 7x7 patches + 1 cls-token
            depth=12,
            heads=8,
            k=64
                )
        model_vit = ViT(
            dim=128,
            image_size= 224,
            patch_size=32,
            num_classes=2,
            transformer=efficient_transformer,
            channels=8,
        )
        model_vit.load_state_dict(torch.load(cfg.model_path_classify))
                # model_vit = model_vit.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # print(next(model_vit.parameters()).device)
        
        model_vit.eval()
        # model_vit = model_vit.to(device)
        model_vit = model_vit.to(torch.device('cpu'))
        pre_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

        with rasterio.open(path_img1) as raster1:
            with rasterio.open(path_img2) as raster2:
                meta = raster1.meta
                meta.update({'count': 1, 'nodata': 0, "dtype": "uint8"})
                height, width = raster1.height, raster1.width
                input_size = cfg.img_size
                stride_size = input_size - input_size // 4
                padding = int((input_size - stride_size) / 2)
                # stride_size = 256
                # padding = 0

                list_coordinates = []
                for start_y in range(0, height, stride_size):
                    for start_x in range(0, width, stride_size):
                        x_off = start_x if start_x == 0 else start_x - padding
                        y_off = start_y if start_y == 0 else start_y - padding

                        end_x = min(start_x + stride_size + padding, width)
                        end_y = min(start_y + stride_size + padding, height)

                        x_count = end_x - x_off
                        y_count = end_y - y_off
                        list_coordinates.append(tuple([x_off, y_off, x_count, y_count, start_x, start_y]))

                with rasterio.open(out_path, 'w+', **meta, compress='lzw') as r:
                    read_lock = threading.Lock()
                    write_lock = threading.Lock()
                
                    def process(coordinates):
                        
                        x_off, y_off, x_count, y_count, start_x, start_y = coordinates
                        read_wd = Window(x_off, y_off, x_count, y_count)
                        with read_lock:
                            values1 = raster1.read(window=read_wd)[0: num_bands]
                            values2 = raster2.read(window=read_wd)[0: num_bands]
                        if raster1.profile["dtype"] == "uint8" and raster2.profile["dtype"] == "uint8":
                            image_detect1 = values1.transpose(1, 2, 0).astype(int)
                            image_detect2 = values2.transpose(1, 2, 0).astype(int)

                        if image_detect1.shape != image_detect2.shape:
                            diff_rows = int(abs(image_detect1.shape[0] - image_detect2.shape[0]))
                            diff_cols = int(abs(image_detect1.shape[1] - image_detect2.shape[1]))

                            if image_detect1.shape[0] > image_detect2.shape[0]:
                                padded_arr1 = np.zeros((image_detect1.shape[0], image_detect1.shape[1], image_detect1.shape[2]))
                                for i in range(num_bands):
                                    padded_arr1[:, :, i] = np.pad(image_detect2[:, :, i], ((0, diff_rows), (0, diff_cols)), mode='constant', constant_values=0)
                                image_detect = np.zeros((image_detect1.shape[0], image_detect1.shape[1], num_bands*2))
                                image_detect[:, :, 0: num_bands] = image_detect1[:, :, 0: num_bands]

                            if image_detect1.shape[0] < image_detect2.shape[0]:
                                padded_arr1 = np.zeros((image_detect2.shape[0], image_detect2.shape[1], image_detect2.shape[2]))
                                for i in range(num_bands):
                                    padded_arr1[:, :, i] = np.pad(image_detect2[:, :, i], ((0, diff_rows), (0, diff_cols)), mode='constant', constant_values=0)
                                image_detect = np.zeros((image_detect2.shape[0], image_detect2.shape[1], num_bands*2))
                                image_detect[:, :, 0: num_bands] = padded_arr1[:, :, 0: num_bands]
                                image_detect[:, :, num_bands: num_bands * 2] = image_detect2[:, :, 0: num_bands]

                            if image_detect1.shape[1] > image_detect2.shape[1]:
                                padded_arr1 = np.zeros((image_detect1.shape[0], image_detect1.shape[1], image_detect1.shape[2]))
                                for i in range(num_bands):
                                    padded_arr1[:, :, i] = np.pad(image_detect2[:, :, i], ((0, diff_rows), (0, diff_cols)), mode='constant', constant_values=0)
                                image_detect = np.zeros((image_detect1.shape[0], image_detect1.shape[1], num_bands * 2))
                                image_detect[:, :, 0: num_bands] = image_detect1[:, :, 0: num_bands]
                                image_detect[:, :, num_bands: num_bands*2] = padded_arr1[:, :, 0: num_bands]

                            if image_detect1.shape[1] < image_detect2.shape[1]:
                                padded_arr1 = np.zeros((image_detect2.shape[0], image_detect2.shape[1], image_detect2.shape[2]))
                                for i in range(num_bands):
                                    padded_arr1[:, :, i] = np.pad(image_detect2[:, :, i], ((0, diff_rows), (0, diff_cols)), mode='constant', constant_values=0)
                                image_detect = np.zeros((image_detect2.shape[0], image_detect2.shape[1], num_bands*2))
                                image_detect[:, :, 0: num_bands] = padded_arr1[:, :, 0: num_bands]
                                image_detect[:, :, num_bands: num_bands*2] = image_detect2[:, :, 0: num_bands]
                        else:
                            image_detect = np.zeros((image_detect1.shape[0], image_detect1.shape[1], num_bands*2))
                            image_detect[:, :, 0: num_bands] = image_detect1[:, :, 0: num_bands]
                            image_detect[:, :, num_bands: num_bands*2] = image_detect2[:, :, 0: num_bands]

                        img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                        mask = np.pad(np.ones((stride_size, stride_size), dtype=np.uint8), ((padding, padding), (padding, padding)))
                        shape = (stride_size, stride_size)

                        if y_count < input_size or x_count < input_size:
                            img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                            mask = np.zeros((input_size, input_size), dtype=np.uint8)
                            if start_x == 0 and start_y == 0:
                                img_temp[(input_size - y_count): input_size, (input_size - x_count): input_size] = image_detect
                                mask[(input_size - y_count): input_size - padding, (input_size - x_count): input_size-padding] = 1
                                shape = (y_count - padding, x_count - padding)
                            elif start_x == 0:
                                img_temp[0: y_count, (input_size - x_count): input_size] = image_detect
                                if y_count == input_size:
                                    mask[padding: y_count - padding, (input_size - x_count): input_size - padding] = 1
                                    shape = (y_count - 2 * padding, x_count - padding)
                                else:
                                    mask[padding: y_count, (input_size - x_count): input_size - padding] = 1
                                    shape = (y_count - padding, x_count - padding)
                            elif start_y == 0:
                                img_temp[(input_size - y_count): input_size, 0: x_count] = image_detect
                                if x_count == input_size:
                                    mask[(input_size - y_count):input_size-padding, padding:x_count-padding] = 1
                                    shape = (y_count - padding, x_count - 2 * padding)
                                else:
                                    mask[(input_size - y_count): input_size - padding, padding: x_count] = 1
                                    shape = (y_count - padding, x_count - padding)
                            else:
                                img_temp[0: y_count, 0: x_count] = image_detect
                                mask[padding: y_count, padding: x_count] = 1
                                shape = (y_count - padding, x_count - padding)

                            image_detect = img_temp
                        img_detect_1 = image_detect[:,:,0:num_bands]
                        img_detect_2 = image_detect[:,:,num_bands:]
                        img_detect_class1 = np.array(img_detect_1)
                        img_detect_class2 = np.array(img_detect_2)
                        mask = (mask != 0)
                   
                        img_detect_1 = TF.to_tensor(img_detect_1)
                        img_detect_2 = TF.to_tensor(img_detect_2)
                        img_detect_1 = img_detect_1/255.0
                        img_detect_2 = img_detect_2/255.0
                    
                        img_detect_1_x = TF.normalize(img_detect_1, mean=[0.5, 0.5, 0.5 , 0.5], std=[0.5, 0.5, 0.5, 0.5])
                        img_detect_2_x = TF.normalize(img_detect_2, mean=[0.5, 0.5, 0.5 , 0.5], std=[0.5, 0.5, 0.5, 0.5])
                 
                        img_detect_1_x = img_detect_1_x.unsqueeze(0).float().to(device)
                        img_detect_2_x = img_detect_2_x.unsqueeze(0).float().to(device)
                        pred = model.net_G(img_detect_1_x,img_detect_2_x)[-1]
                        predict = torch.argmax(pred, dim=1, keepdim=True)
                        pred_vis = predict 
                        
                        predict_img = pred_vis[0].cpu().numpy()
                        predict_img = predict_img[0,:,:]
                        y = predict_img[mask].reshape(shape)

                        img_detect_class1 = Image.fromarray(img_detect_class1.astype(np.uint8))
                        img_detect_class2 = Image.fromarray(img_detect_class2.astype(np.uint8))
                        img_detect_class1 = pre_transforms(img_detect_class1)
                        img_detect_class2 = pre_transforms(img_detect_class2)
                        img_detect_class = torch.cat((img_detect_class1,img_detect_class2), dim=0)
                        img_detect_class = img_detect_class.unsqueeze(0)
                        with torch.no_grad():
                            output = model_vit(img_detect_class)
                            _, predicted = torch.max(output, 1)
                        
                        if predicted.item() == 0:
                           
                            y[y!=0] =0
                        else:
                            y = y
                        with write_lock:
                            r.write(y[np.newaxis, ...], window=Window(start_x, start_y, shape[1], shape[0]))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                        results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))

        return out_path

def predict_shadow( path_image1, path_image2, path_predict,cfg,path_model):
    cfg = cfg
    model = tf.keras.models.load_model(path_model)
    num_bands = cfg.num_bands
    with rasterio.open(path_image1) as raster1:
        with rasterio.open(path_image2) as raster2:
            meta = raster1.meta
            meta.update({'count': 1, 'nodata': 0, "dtype": "uint8"})
            height, width = raster1.height, raster1.width
            input_size = cfg.img_size
            stride_size = input_size - input_size // 4
            padding = int((input_size - stride_size) / 2)

            list_coordinates = []
            for start_y in range(0, height, stride_size):
                for start_x in range(0, width, stride_size):
                    x_off = start_x if start_x == 0 else start_x - padding
                    y_off = start_y if start_y == 0 else start_y - padding

                    end_x = min(start_x + stride_size + padding, width)
                    end_y = min(start_y + stride_size + padding, height)

                    x_count = end_x - x_off
                    y_count = end_y - y_off
                    list_coordinates.append(tuple([x_off, y_off, x_count, y_count, start_x, start_y]))

            with rasterio.open(path_predict, 'w+', **meta, compress='lzw') as r:
                read_lock = threading.Lock()
                write_lock = threading.Lock()

                def process(coordinates):
                    x_off, y_off, x_count, y_count, start_x, start_y = coordinates
                    read_wd = Window(x_off, y_off, x_count, y_count)
                    with read_lock:
                        values1 = raster1.read(window=read_wd)[0: num_bands]
                        values2 = raster2.read(window=read_wd)[0: num_bands]
                    if raster1.profile["dtype"] == "uint8" and raster2.profile["dtype"] == "uint8":
                        image_detect1 = values1.transpose(1, 2, 0).astype(int)
                        image_detect2 = values2.transpose(1, 2, 0).astype(int)

                    if image_detect1.shape != image_detect2.shape:
                        diff_rows = int(abs(image_detect1.shape[0] - image_detect2.shape[0]))
                        diff_cols = int(abs(image_detect1.shape[1] - image_detect2.shape[1]))

                        if image_detect1.shape[0] > image_detect2.shape[0]:
                            padded_arr1 = np.zeros((image_detect1.shape[0], image_detect1.shape[1], image_detect1.shape[2]))
                            for i in range(num_bands):
                                padded_arr1[:, :, i] = np.pad(image_detect2[:, :, i], ((0, diff_rows), (0, diff_cols)), mode='constant', constant_values=0)
                            image_detect = np.zeros((image_detect1.shape[0], image_detect1.shape[1], num_bands*2))
                            image_detect[:, :, 0: num_bands] = image_detect1[:, :, 0: num_bands]

                        if image_detect1.shape[0] < image_detect2.shape[0]:
                            padded_arr1 = np.zeros((image_detect2.shape[0], image_detect2.shape[1], image_detect2.shape[2]))
                            for i in range(num_bands):
                                padded_arr1[:, :, i] = np.pad(image_detect2[:, :, i], ((0, diff_rows), (0, diff_cols)), mode='constant', constant_values=0)
                            image_detect = np.zeros((image_detect2.shape[0], image_detect2.shape[1], num_bands*2))
                            image_detect[:, :, 0: num_bands] = padded_arr1[:, :, 0: num_bands]
                            image_detect[:, :, num_bands: num_bands * 2] = image_detect2[:, :, 0: num_bands]

                        if image_detect1.shape[1] > image_detect2.shape[1]:
                            padded_arr1 = np.zeros((image_detect1.shape[0], image_detect1.shape[1], image_detect1.shape[2]))
                            for i in range(num_bands):
                                padded_arr1[:, :, i] = np.pad(image_detect2[:, :, i], ((0, diff_rows), (0, diff_cols)), mode='constant', constant_values=0)
                            image_detect = np.zeros((image_detect1.shape[0], image_detect1.shape[1], num_bands * 2))
                            image_detect[:, :, 0: num_bands] = image_detect1[:, :, 0: num_bands]
                            image_detect[:, :, num_bands: num_bands*2] = padded_arr1[:, :, 0: num_bands]

                        if image_detect1.shape[1] < image_detect2.shape[1]:
                            padded_arr1 = np.zeros((image_detect2.shape[0], image_detect2.shape[1], image_detect2.shape[2]))
                            for i in range(num_bands):
                                padded_arr1[:, :, i] = np.pad(image_detect2[:, :, i], ((0, diff_rows), (0, diff_cols)), mode='constant', constant_values=0)
                            image_detect = np.zeros((image_detect2.shape[0], image_detect2.shape[1], num_bands*2))
                            image_detect[:, :, 0: num_bands] = padded_arr1[:, :, 0: num_bands]
                            image_detect[:, :, num_bands: num_bands*2] = image_detect2[:, :, 0: num_bands]
                    else:
                        image_detect = np.zeros((image_detect1.shape[0], image_detect1.shape[1], num_bands*2))
                        image_detect[:, :, 0: num_bands] = image_detect1[:, :, 0: num_bands]
                        image_detect[:, :, num_bands: num_bands*2] = image_detect2[:, :, 0: num_bands]

                    img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                    mask = np.pad(np.ones((stride_size, stride_size), dtype=np.uint8), ((padding, padding), (padding, padding)))
                    shape = (stride_size, stride_size)

                    if y_count < input_size or x_count < input_size:
                        img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                        mask = np.zeros((input_size, input_size), dtype=np.uint8)
                        if start_x == 0 and start_y == 0:
                            img_temp[(input_size - y_count): input_size, (input_size - x_count): input_size] = image_detect
                            mask[(input_size - y_count): input_size - padding, (input_size - x_count): input_size-padding] = 1
                            shape = (y_count - padding, x_count - padding)
                        elif start_x == 0:
                            img_temp[0: y_count, (input_size - x_count): input_size] = image_detect
                            if y_count == input_size:
                                mask[padding: y_count - padding, (input_size - x_count): input_size - padding] = 1
                                shape = (y_count - 2 * padding, x_count - padding)
                            else:
                                mask[padding: y_count, (input_size - x_count): input_size - padding] = 1
                                shape = (y_count - padding, x_count - padding)
                        elif start_y == 0:
                            img_temp[(input_size - y_count): input_size, 0: x_count] = image_detect
                            if x_count == input_size:
                                mask[(input_size - y_count):input_size-padding, padding:x_count-padding] = 1
                                shape = (y_count - padding, x_count - 2 * padding)
                            else:
                                mask[(input_size - y_count): input_size - padding, padding: x_count] = 1
                                shape = (y_count - padding, x_count - padding)
                        else:
                            img_temp[0: y_count, 0: x_count] = image_detect
                            mask[padding: y_count, padding: x_count] = 1
                            shape = (y_count - padding, x_count - padding)

                        image_detect = img_temp

                    mask = (mask != 0)

                    if np.count_nonzero(image_detect) > 0:
                        if len(np.unique(image_detect)) <= 2:
                            pass
                        else:
                            y_pred = model.predict(image_detect[np.newaxis, ...] / 255.)
                            y_pred = np.array(y_pred)
                            y_pred = (y_pred[0, 0, ..., 0] > 0.45).astype(np.uint8)
                            y = y_pred[mask].reshape(shape)
                            with write_lock:
                                r.write(y[np.newaxis, ...], window=Window(start_x, start_y, shape[1], shape[0]))
                with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                    results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))

    return path_predict   

