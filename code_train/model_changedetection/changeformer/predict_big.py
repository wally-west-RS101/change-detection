from argparse import ArgumentParser

import utils
import torch
from models.basic_model import CDEvaluator
import torchvision.transforms.functional as TF
import os
import numpy as np
import rasterio
from rasterio.windows import Window
import threading
from tqdm import tqdm
import concurrent.futures
import warnings
import cv2
import os
import glob
import torchvision.transforms as transforms
"""
quick start

sample files in ./samples

save prediction files in the ./samples/predict

"""


def get_args():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--project_name', default='KET_QUA_MUILTI', type=str)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoint_root', default='/home/skymap/data/Newmodel_cd/change_multi/ChangeFormer/checkpoints', type=str)
    parser.add_argument('--output_folder', default='samples_LEVIR/predict_CD_ChangeFormerV6_v2122_4b', type=str)

    # data
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR', type=str)

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="test", type=str)
    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=4, type=int)
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--net_G', default='ChangeFormerV6', type=str,
                        help='ChangeFormerV6 | CD_SiamUnet_diff | SiamUnet_conc | Unet | DTCDSCN | base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')
    parser.add_argument('--checkpoint_name', default='best_ckpt_691_3class.pt', type=str)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()
    utils.get_device(args)
    device = torch.device("cuda:%s" % args.gpu_ids[0]
                          if torch.cuda.is_available() and len(args.gpu_ids)>0
                        else "cpu")
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    print(args.checkpoint_dir)
    os.makedirs(args.output_folder, exist_ok=True)

    log_path = os.path.join(args.output_folder, 'log_vis.txt')
    model = CDEvaluator(args)
    model.load_checkpoint(args.checkpoint_name)
    print(args.checkpoint_name)
    model.eval()
    # data_loader = utils.get_loader(args.data_name, img_size=args.img_size,
    #                                batch_size=args.batch_size,
    #                                split=args.split, is_train=False)
    path_dir_before_image = '/home/skymap/data/Bahrain_change/allqc/BAOCAO_25_3/set_2324_img_cut/set23'
    path_dir_after_image = '/home/skymap/data/Bahrain_change/allqc/BAOCAO_25_3/set_2324_img_cut/set24'
    out_dir = '/home/skymap/data/Bahrain_change/allqc/kq_3class/set_2324__epoc_691'
    path_da_predict = '/home/skymap/data/Bahrain_change/allqc/kq_3class/set_2324__epoc_691'
    list_path_dapredict = []
    for fp_path_da_pre in glob.glob(os.path.join(path_da_predict,'*.tif')):
        f_name = os.path.basename(fp_path_da_pre)
        name , _ = os.path.splitext(f_name)
        list_path_dapredict.append(name)
    print('da_pred',len(list_path_dapredict))
    for fp_bf in glob.glob(os.path.join(path_dir_before_image,"*.tif")):
        full_name = os.path.basename(fp_bf)
        name_bf , _ = os.path.splitext(full_name)
        name_af = name_bf.replace('23','24')
        if name_af in list_path_dapredict:
            print('da_predict',name_af)
            continue
        fp_after = os.path.join(path_dir_after_image,f'{name_af}.tif') 
        out_path_predict = os.path.join(out_dir,f'{name_af}.tif')
        print('dang predict anh trc : ',fp_bf)
        print('dang predict anh sau : ',fp_after)
        print('dau ra du kien: ',out_path_predict)
        num_bands = 4
        with rasterio.open(fp_bf) as raster1:
            with rasterio.open(fp_after) as raster2:
                meta = raster1.meta
                meta.update({'count': 1, 'nodata': 0, "dtype": "uint8"})
                height, width = raster1.height, raster1.width
                input_size = args.img_size
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

                with rasterio.open(out_path_predict, 'w+', **meta, compress='lzw') as r:
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
                        # img1 = img_detect_1[:,:,0:3]
                        # img2 = img_detect_2[:,:,0:3]
                        # img1 = img1.astype(np.uint8)
                        # img2 = img2.astype(np.uint8)
                        # cv2.imwrite(f'/home/skm/SKM16/DAO/test_changef/check_pre/A/{x_off}_{y_off}.png',img1)
                        # cv2.imwrite(f'/home/skm/SKM16/DAO/test_changef/check_pre/B/{x_off}_{y_off}.png',img2)
                        
                        
                        # print('kkkk',img_detect_2.shape)
                        mask = (mask != 0)
                        # img = img_detect_2
                        # img = img.astype(np.uint8)
                        # print(np.unique(img))
                        # print(img.shape)
                        # from matplotlib import pyplot as plt
                        # plt.imshow(img[:,:,0:3])
                        # plt.show()
                    
                        # imgs = [TF.to_tensor(img) for img in imgs]
                        img_detect_1 = TF.to_tensor(img_detect_1)
                        img_detect_2 = TF.to_tensor(img_detect_2)
                        img_detect_1 = img_detect_1/255.0
                        img_detect_2 = img_detect_2/255.0
                        # print('nnnnnnnnnnnn',torch.unique(img_detect_1))
                        # mean=[0.5, 0.5, 0.5 , 0.5] 
                        # std=[0.5, 0.5, 0.5, 0.5]
                        img_detect_1_x = TF.normalize(img_detect_1, mean=[0.5, 0.5, 0.5 , 0.5], std=[0.5, 0.5, 0.5, 0.5])
                        img_detect_2_x = TF.normalize(img_detect_2, mean=[0.5, 0.5, 0.5 , 0.5], std=[0.5, 0.5, 0.5, 0.5])
                        # normalize_transform = transforms.Normalize(mean=mean, std=std)
                        # img_detect_1_x = normalize_transform(img_detect_1).to(device)
                        # img_detect_2_x = normalize_transform(img_detect_2).to(device)
                        # img_detect_1_x = img_detect_1/255.0*2 -1
                        # img_detect_2_x = img_detect_2/255.0*2 -1
                        img_detect_1_x = img_detect_1_x.unsqueeze(0).float().to(device)
                        img_detect_2_x = img_detect_2_x.unsqueeze(0).float().to(device)
                        # print(img_detect_1_x.shape)
                        # print('kkkkk',torch.unique(img_detect_1_x))
                        pred = model.net_G(img_detect_1_x,img_detect_2_x)[-1]
                        predict = torch.argmax(pred, dim=1, keepdim=True)
                        pred_vis = predict 
                        
                        predict_img = pred_vis[0].cpu().numpy()
                        predict_img = predict_img[0,:,:]
                        y = predict_img[mask].reshape(shape)
                        # imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        #         for img in imgs]
                        
                        # imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5 , 0.5], std=[0.5, 0.5, 0.5, 0.5])
                        #         for img in imgs]

                        # if np.count_nonzero(image_detect) > 0:
                        #     if len(np.unique(image_detect)) <= 2:
                        #         pass
                        #     else:
                        #         y_pred = model.predict(image_detect[np.newaxis, ...] / 255.)
                        #         y_pred = np.array(y_pred)
                        #         y_pred = (y_pred[0, 0, ..., 0] > 0.45).astype(np.uint8)
                        #         y = y_pred[mask].reshape(shape)
                        #         # y = Morphology(y)
                        with write_lock:
                            r.write(y[np.newaxis, ...], window=Window(start_x, start_y, shape[1], shape[0]))
                    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                        results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))

