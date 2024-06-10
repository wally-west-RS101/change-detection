import numpy as np
import rasterio
from rasterio.windows import Window
import threading
from tqdm import tqdm
import concurrent.futures
import warnings
import cv2
import os
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import glob

warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

set_session(tf.compat.v1.Session(config=config))

num_bands = 4
size = 512


def predict_farm(model, path_image1, path_image2, path_predict, size=512):

    with rasterio.open(path_image1) as raster1:
        with rasterio.open(path_image2) as raster2:
            meta = raster1.meta
            meta.update({'count': 1, 'nodata': 0, "dtype": "uint8"})
            height, width = raster1.height, raster1.width
            input_size = size
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
                            y = Morphology(y)
                            with write_lock:
                                r.write(y[np.newaxis, ...], window=Window(start_x, start_y, shape[1], shape[0]))
                with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
                    results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))


def Morphology(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img = cv2.dilate(image, kernel, iterations=1)
    img = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
    return img


def bo_file_tao_ra_muon_nhat_trong_list(list_fp):
    time_max = 0
    fp_break = 's'
    for fp in list_fp:
        time_create = os.path.getmtime(fp)
        if time_create > time_max:
            time_max = time_create
            fp_break = fp
    list_fp.remove(fp_break)
    return list_fp


def keep_list_fp_dont_have_list_eliminate(list_have_all, list_eliminate):
    list_eliminate = [os.path.basename(fp) for fp in list_eliminate]
    if list_eliminate:
        list_keep = []
        for fp in list_have_all:
            if os.path.basename(fp) not in list_eliminate:
                list_keep.append(fp)
        return list_keep
    else:
        return list_have_all


if __name__ == "__main__":
    model_path = r'/home/skymap/data/Bahrain_change/LAM LAI/LABEL/WEIGHT_MOI/other/u2net_512_other_9_6_epoch_9_model.h5'
    dir_img_1 = r"/home/skymap/data/Bahrain_change/SET14/t13_regist14/gtif"
    dir_img_2 = r'/home/skymap/data/Bahrain_change/SET14/T14_rgb'
    dir_out = r"/home/skymap/data/Bahrain_change/SET14/out/other"
    os.makedirs(dir_out, exist_ok=True)
    list_img1 = glob.glob(os.path.join(dir_img_1, '*.tif'))
    list_img2 = glob.glob(os.path.join(dir_img_2, '*.tif'))
    list_img1.sort()
    list_img2.sort()
    print(len(list_img1), 'all')
    count = 0
    model_farm = tf.keras.models.load_model(model_path)
    size = 512
    for input_path1, input_path2 in zip(list_img1, list_img2):
        print('dang predict anh:', input_path1)
        print('dang predict anh:', input_path2)
        print('count', count)
        output_path = os.path.join(dir_out, os.path.basename(input_path1))
        predict_farm(model_farm, input_path1, input_path2, output_path, size)
        count += 1
