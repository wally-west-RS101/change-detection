# from curses import has_ic
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
from rio_tiler.io import COGReader
from tensorflow.compat.v1.keras.backend import set_session
warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))
import glob
def get_quantile_schema(img):
    qt_scheme = []
    try:
        with COGReader(img) as cog:
            stats = cog.stats()
            for _, value in stats.items():
                qt_scheme.append({
                    'p2': value['percentiles'][0],
                    'p98': value['percentiles'][1],
                })
    except:
        with COGReader(img) as cog:
            stats = cog.statistics()
            for _, value in stats.items():
                qt_scheme.append({
                    'p2': value['percentile_2'],
                    'p98': value['percentile_98'],
                })
    with rasterio.open(img) as r:
        num_band = r.count
        for chanel in range(1,num_band+1):
            data = r.read(chanel).astype(np.float16)
            data[data==0] = np.nan
            qt_scheme.append({
                'p2': np.nanpercentile(data, 2),
                'p98': np.nanpercentile(data, 98),
            })
    return qt_scheme
def get_quantile_schema(img):
    qt_scheme = []
    with COGReader(img) as cog:
        stats = cog.stats()
        for _, value in stats.items():
            qt_scheme.append({
                'p2': value['percentile_2'],
                'p98': value['percentile_98'],
            })
    return qt_scheme
def batch_split(iterable, n=1):

    new_list = []
    l = len(iterable)
    for ndx in range(0, l, n):
        new_list.append(iterable[ndx:min(ndx + n, l)])
    return new_list


def predict_farm(model, path_image1,path_image2, path_predict, cfd, num_bands, size=512):
    print('cfd',cfd)
    batch_size = 8
    # print(111111111111111111111112222222222222)
    # qt_scheme = get_quantile_schema(path_image)
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
            # print(22222222222222222222222222222)
            with rasterio.open(path_predict, 'w+', **meta, compress='lzw') as r:
                read_lock = threading.Lock()
                write_lock = threading.Lock()

                def process(coordinates):
                    list_image_detect = []
                    list_mask = []
                    list_shape = []
                    with read_lock:
                        # print(1)
                        for coord in coordinates:
                            x_off, y_off, x_count, y_count, start_x, start_y = coord
                            read_wd = Window(x_off, y_off, x_count, y_count)
                            # print(33333333333333333333333333333)
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
                                    image_detect[:, :, num_bands: num_bands * 2] = padded_arr1[:, :, 0: num_bands]

                                if image_detect1.shape[0] < image_detect2.shape[0]:
                                    padded_arr1 = np.zeros((image_detect2.shape[0], image_detect2.shape[1], image_detect2.shape[2]))
                                    for i in range(num_bands):
                                        padded_arr1[:, :, i] = np.pad(image_detect1[:, :, i], ((0, diff_rows), (0, diff_cols)), mode='constant', constant_values=0)
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
                                        padded_arr1[:, :, i] = np.pad(image_detect1[:, :, i], ((0, diff_rows), (0, diff_cols)), mode='constant', constant_values=0)
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

                            # print(1111111111111111111)
                            mask = (mask != 0)
                            list_image_detect.append(cv2.resize(image_detect.astype(np.uint8), (size,size), interpolation = cv2.INTER_AREA))
                            list_mask.append(mask)
                            list_shape.append(shape)
                        numpy_list_image_detect = np.array(list_image_detect)
                        if np.count_nonzero(numpy_list_image_detect) > 0:
                            if len(np.unique(numpy_list_image_detect)) <= 2:
                                pass
                            else:
                                # resized = cv2.resize(image_detect.astype(np.uint8), (256,256), interpolation = cv2.INTER_AREA)
                                y_pred = model.predict(numpy_list_image_detect/255)
                                y_pred = np.array(y_pred).transpose(1,0,2,3,4)
                            # print(y_pred.shape)
                                with write_lock:
                                    for i in range(len(coordinates)):
                                        x_off, y_off, x_count, y_count, start_x, start_y = coordinates[i]
                                        

                                        y_pred0 = y_pred[i]
                                        y_pred0 = (y_pred0[0,...,0] > cfd).astype(np.uint8)
                                        mask0= list_mask[i]
                                        shape0 = list_shape[i]
                                    # y_pred = np.argmax(y_pred,axis=-1)
                                    # print(y_pred.shape)
                                        y_pred0 = (cv2.resize(y_pred0, (size,size), interpolation = cv2.INTER_AREA)>0.5).astype(np.uint8)
                                        y = y_pred0[mask0].reshape(shape0)

                                        
                                        r.write(y[np.newaxis,...], window=Window(start_x, start_y, shape0[1], shape0[0]))

                chunks = batch_split(list_coordinates,n=batch_size)
                # print(chunks[884:])
                for coordsss in tqdm(chunks):
                    # print(coordsss)
                    process(coordsss)
            # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            #     results = list(tqdm(executor.map(process, chunks), total=len(chunks)))


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

from skimage.morphology import skeletonize
def raster_to_vector(path_in, path_out, threshold_distance, threshold_connect):
    print('start convert raster to vector ...')
    with rasterio.open(path_in) as inds:
        data = inds.read()[0]
        transform = inds.transform
        projstr = inds.crs.to_string()
        
    data = Morphology(data)
    data = remove_small_holes(data.astype(bool), area_threshold=77)
    data = remove_small_objects(data, min_size=77)
    skeleton = skeletonize(data.astype(np.uint8))
    
    # Vectorization.save_polygon(np.pad(skeleton, pad_width=1).astype(np.intc), threshold_distance, threshold_connect, transform, projstr, path_out)
    print("Done!!!")
    
if __name__=="__main__":
  
    threshold_distance = 3 #ngưỡng làm mượt polygon
    threshold_connect = 3 #5 #ngưỡng nối điểm gần nhau


    model_path_building = r'/home/skymap/data/Bahrain_change/allqc/weights_bong/u2net_shadow_v3_new_24t2__56_modelv1.h5'
    # model_path_vege = r'/home/skymap/data/Bahrain_change/Improve_model/train_improve_vege/w/u2net_512_improve_12_11_v1_11_model.h5'
    # model_path_other = r'/home/skymap/data/Bahrain_change/LAM LAI/LABEL/WEIGHT_MOI/other/u2net_512_other_9_6_epoch_9_model.h5'
    dir_img_1 = r"/home/skymap/data/Bahrain_change/set18/T17_RGB"
    dir_img_2 = r'/home/skymap/data/Bahrain_change/set18/T18_RGB'
    dir_out_building = r"/home/skymap/data/Bahrain_change/allqc/BAOCAO_4_3/bong_1718"
    # dir_out_vege = r"/home/skymap/data/Bahrain_change/set23/out/vege"
    # dir_out_other = r"/home/skymap/data/Bahrain_change/set23/out/other"
    # path_da_predict = '/home/skymap/data/Bahrain_change/set21/out/vege'
    # list_da_predict = glob.glob(os.path.join(path_da_predict,'*.tif'))
    # list_name_predict = []
    # for image_pre in list_da_predict:
    #     name = os.path.basename(image_pre)
    #     list_name_predict.append(name)
    os.makedirs(dir_out_building, exist_ok=True)
    # os.makedirs(dir_out_vege, exist_ok=True)
    # os.makedirs(dir_out_other, exist_ok=True)
    list_img1 = glob.glob(os.path.join(dir_img_1, '*.tif'))
    list_img2 = glob.glob(os.path.join(dir_img_2, '*.tif'))
    list_img1.sort()
    list_img2.sort()
    print(len(list_img1), 'all')
    count = 0
    model_bd = tf.keras.models.load_model(model_path_building)
    # model_vege = tf.keras.models.load_model(model_path_vege)
    # model_other = tf.keras.models.load_model(model_path_other)
    model_bd.trainable = False
    # model_vege.trainable = False
    # model_other.trainable = False
    size = 512
    cfd_building = 0.4
    cfd_vege = 0.4
    cfd_other = 0.4
    print('==============================building===========================================')
    for input_path1, input_path2 in zip(list_img1, list_img2):
        print('dang predict anh:', input_path1)
        print('dang predict anh:', input_path2)
        print('count', count)
        output_path = os.path.join(dir_out_building, os.path.basename(input_path1))
        predict_farm(model_bd, input_path1, input_path2, output_path, cfd_building,num_bands=4, size=size)
        count += 1
    # print('==============================vege===========================================')
    # for input_path1, input_path2 in zip(list_img1, list_img2):
    #     name1 = os.path.basename(input_path1)
    #     # if name1 in list_name_predict:
    #     #     print('11111111111111111111111111111')
    #     #     continue
    #     print('dang predict anh:', input_path1)
    #     print('dang predict anh:', input_path2)
    #     print('count', count)
    #     output_path = os.path.join(dir_out_vege, os.path.basename(input_path1))
    #     predict_farm(model_vege, input_path1, input_path2, output_path, cfd_vege, num_bands=4, size=size)
    #     count += 1
    # print('==============================other===========================================')
    # for input_path1, input_path2 in zip(list_img1, list_img2):
       
    #     print('dang predict anh:', input_path1)
    #     print('dang predict anh:', input_path2)
    #     print('count', count)
      
        
            
    #     output_path = os.path.join(dir_out_other, os.path.basename(input_path1))
    #     predict_farm(model_other, input_path1, input_path2, output_path, cfd_other, num_bands=4, size=size)
    #     count += 1
        
        