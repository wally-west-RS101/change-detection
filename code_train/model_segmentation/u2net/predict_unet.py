import numpy as np
import rasterio
import argparse
from rasterio.windows import Window
import threading
from tqdm import tqdm
import concurrent.futures
import warnings, cv2, os
import tensorflow as tf
# import Vectorization
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
# from rio_tiler.io import COGReader
from tensorflow.compat.v1.keras.backend import set_session
import os, glob

warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))
num_bands = 4
def get_quantile_schema(img):
    pass

# def get_quantile_schema(img):
#     qt_scheme = []
#     try:
#         with COGReader(img) as cog:
#             stats = cog.stats()
#             for _, value in stats.items():
#                 qt_scheme.append({
#                     'p2': value['percentiles'][0],
#                     'p98': value['percentiles'][1],
#                 })
#     except:
#         with COGReader(img) as cog:
#             stats = cog.statistics()
#             for _, value in stats.items():
#                 qt_scheme.append({
#                     'p2': value['percentile_2'],
#                     'p98': value['percentile_98'],
#                 })
# #     with rasterio.open(img) as r:
# #         num_band = r.count
# #         for chanel in range(1,num_band+1):
# #             data = r.read(chanel).astype(np.float16)
# #             data[data==0] = np.nan
# #             qt_scheme.append({
# #                 'p2': np.nanpercentile(data, 2),
# #                 'p98': np.nanpercentile(data, 98),
# #             })
# #     # print(qt_scheme)
#     return qt_scheme
# # def get_quantile_schema(img):
# #     qt_scheme = []
# #     with COGReader(img) as cog:
# #         stats = cog.stats()
# #         for _, value in stats.items():
# #             qt_scheme.append({
# #                 'p2': value['percentile_2'],
# #                 'p98': value['percentile_98'],
# #             })
# #     return qt_scheme

def predict_farm(model, path_image, path_predict, size=256):
    qt_scheme = get_quantile_schema(path_image)
    with rasterio.open(path_image) as raster:
        meta = raster.meta
        meta.update({'count': 1, 'nodata': 0,"dtype":"uint8"})
        height, width = raster.height, raster.width
        input_size = size
        stride_size = input_size - input_size //4
        padding = int((input_size - stride_size) / 2)

        list_coordinates = []
        for start_y in range(0, height, stride_size):
            for start_x in range(0, width, stride_size):
                x_off = start_x if start_x==0 else start_x - padding
                y_off = start_y if start_y==0 else start_y - padding

                end_x = min(start_x + stride_size + padding, width)
                end_y = min(start_y + stride_size + padding, height)

                x_count = end_x - x_off
                y_count = end_y - y_off
                list_coordinates.append(tuple([x_off, y_off, x_count, y_count, start_x, start_y]))
        with rasterio.open(path_predict,'w+', **meta, compress='lzw') as r:
            read_lock = threading.Lock()
            write_lock = threading.Lock()

            def process(coordinates):
                x_off, y_off, x_count, y_count, start_x, start_y = coordinates
                read_wd = Window(x_off, y_off, x_count, y_count)
                with read_lock:
                    values = raster.read(window=read_wd)[0:num_bands]
                if raster.profile["dtype"]=="uint8":
                    # print('vao')
                    image_detect = values.transpose(1,2,0).astype(int)
                    
                else:
                    datas = []
                    for chain_i in range(3):
                        band_qt = qt_scheme[chain_i]
                        band = values[chain_i]
                        cut_nor = np.interp(band, (band_qt.get('p2'), band_qt.get('p98')), (1, 255)).astype(int)
                        datas.append(cut_nor)
                    image_detect = np.array(datas).transpose(1,2,0)
                img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                mask = np.pad(np.ones((stride_size, stride_size), dtype=np.uint8), ((padding, padding),(padding, padding)))
                shape = (stride_size, stride_size)
                if y_count < input_size or x_count < input_size:
                    img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                    mask = np.zeros((input_size, input_size), dtype=np.uint8)
                    if start_x == 0 and start_y == 0:
                        img_temp[(input_size - y_count):input_size, (input_size - x_count):input_size] = image_detect
                        mask[(input_size - y_count):input_size-padding, (input_size - x_count):input_size-padding] = 1
                        shape = (y_count-padding, x_count-padding)
                    elif start_x == 0:
                        img_temp[0:y_count, (input_size - x_count):input_size] = image_detect
                        if y_count == input_size:
                            mask[padding:y_count-padding, (input_size - x_count):input_size-padding] = 1
                            shape = (y_count-2*padding, x_count-padding)
                        else:
                            mask[padding:y_count, (input_size - x_count):input_size-padding] = 1
                            shape = (y_count-padding, x_count-padding)
                    elif start_y == 0:
                        img_temp[(input_size - y_count):input_size, 0:x_count] = image_detect
                        if x_count == input_size:
                            mask[(input_size - y_count):input_size-padding, padding:x_count-padding] = 1
                            shape = (y_count-padding, x_count-2*padding)
                        else:
                            mask[(input_size - y_count):input_size-padding, padding:x_count] = 1
                            shape = (y_count-padding, x_count-padding)
                    else:
                        img_temp[0:y_count, 0:x_count] = image_detect
                        mask[padding:y_count, padding:x_count] = 1
                        shape = (y_count-padding, x_count-padding)

                    image_detect = img_temp
                #print('image',image_detect.shape)
                mask = (mask!=0)
                
                # print(image_detect.shape, 'eeeee')

                if np.count_nonzero(image_detect) > 0:
                    if len(np.unique(image_detect)) <= 2:
                        pass
                    else:
                        # print('ttttttttttt')
                        y_pred = model.predict(image_detect[np.newaxis,...]/255.)
                       
                        #them
                        y_pred = np.array(y_pred)
                        #print('y_pred',y_pred.shape)
                        #goc
                        y_pred = (y_pred[0,...,0] > 0.5).astype(np.uint8)
                        #muilti bands
                        #y_pred = (y_pred[0,0,...,0] > 0.5).astype(np.uint8)
                        #print('y_pred',y_pred)
                        #y_pred = (y_pred[0,:,:,0] > 0.5).astype(np.uint8)
                        #print(y_pred.shape)
                        # print(shape)
                        #print(y_pred.shape)
                        y = y_pred[mask].reshape(shape)
                        

                        with write_lock:
                            r.write(y[np.newaxis,...], window=Window(start_x, start_y, shape[1], shape[0]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))


def Morphology(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # dilation
    # img = cv2.dilate(data,kernel,iterations = 1)
    # opening
    #     img = cv2.morphologyEx(data, cv2.MORPH_OPEN, kernel)
    # for i in range(10):
    #     img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)
    # closing
    #     for _ in range(2):
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

if __name__=="__main__":
    model_path = r'/home/skymap/data/WATER_log_resize/training/train_unet/log/unet_scanmap_val_weights_last_model.h5'
    dir_img = r"/home/skymap/data/WATER_log_resize/image_goc_resize"
    dir_out = r"/home/skymap/data/WATER_log_resize/image_goc_resize/out_unet"
    #dir_name = r"/home/skymap/big_data/Dao_work_space/OpenLandstraindata/code/name.txt"
    
    os.makedirs(dir_out, exist_ok=True)
    list_img = glob.glob(os.path.join(dir_img,'*.tif'))
    print(len(list_img), 'all')
    

    
    
    model_farm = tf.keras.models.load_model(model_path)
    size = 256
    # name = []
    # with open(dir_name,'r') as f:
    #     name = [line.rstrip('\n') for line in f]

    # for input_path in list_img:
    #     if (os.path.basename(input_path) in name):
    #         print(f'xu li anh {os.path.basename(input_path)}')
    #         output_path = os.path.join(dir_out, os.path.basename(input_path))
    for input_path in list_img:
        output_path = os.path.join(dir_out,os.path.basename(input_path))
        predict_farm(model_farm, input_path, output_path, size)
            
                


   