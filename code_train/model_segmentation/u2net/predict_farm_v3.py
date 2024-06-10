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


def predict_farm(model, path_image, path_predict, size=256):
    batch_size = 16
    # qt_scheme = get_quantile_schema(path_image)
    with rasterio.open(path_image) as raster:
        meta = raster.meta
        meta.update({'count': 1, 'nodata': 0,"dtype":"uint8"})
        height, width = raster.height, raster.width
        input_size = 480
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
        with rasterio.open(path_predict,'r+', **meta, compress='lzw') as r:
            read_lock = threading.Lock()
            write_lock = threading.Lock()

            def process(coordinates):
                list_image_detect = []
                list_mask = []
                list_shape = []
                list_resize = []
                with read_lock:
                    for coord in coordinates:
                        
                        x_off, y_off, x_count, y_count, start_x, start_y = coord
                        read_wd = Window(x_off, y_off, x_count, y_count)
                        
                        values = raster.read([1,2,3],window=read_wd)
                        if raster.profile["dtype"]=="uint8":
                            image_detect = values.transpose(1,2,0).astype(int)
                        else:
                            datas =  values/255  # 8192
                            idx_over = np.where(datas > 1)
                            datas[idx_over] = 1 
                            # datas = []
                            # for chain_i in range(8):
                            #     band_qt = qt_scheme[chain_i]
                            #     band = values[chain_i]
                            #     cut_nor = np.interp(band, (band_qt.get('p2'), band_qt.get('p98')), (1, 255)).astype(int)
                            #     datas.append(cut_nor)
                            image_detect = np.array(datas).transpose(1,2,0)
                        img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                        mask = np.pad(np.ones((stride_size, stride_size), dtype=np.uint8),
                                    ((padding, padding), (padding, padding)))
                        shape = (stride_size, stride_size)
                        if y_count < input_size or x_count < input_size:
                            img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                            mask = np.zeros((input_size, input_size), dtype=np.uint8)
                            if start_x == 0 and start_y == 0:
                                img_temp[(input_size - y_count):input_size,
                                (input_size - x_count):input_size] = image_detect

                                mask[(input_size - y_count):input_size - padding,
                                (input_size - x_count):(input_size - padding)] = 1
                                shape = (y_count - padding, x_count - padding)
                            elif start_x == 0:
                                img_temp[0:y_count, (input_size - x_count):input_size] = image_detect
                                if y_count == input_size:
                                    mask[padding:y_count - padding, (input_size - x_count):input_size - padding] = 1
                                    shape = (y_count - 2 * padding, x_count - padding)
                                else:
                                    mask[padding:y_count, (input_size - x_count):input_size - padding] = 1
                                    shape = (y_count - padding, x_count - padding)
                            elif start_y == 0:
                                img_temp[(input_size - y_count):input_size, 0:x_count] = image_detect
                                if x_count == input_size:
                                    mask[(input_size - y_count):input_size - padding, padding:x_count - padding] = 1
                                    shape = (y_count - padding, x_count - 2 * padding)
                                else:
                                    mask[(input_size - y_count):input_size - padding, padding:x_count] = 1
                                    shape = (y_count - padding, x_count - padding)
                            else:
                                img_temp[0:y_count, 0:x_count] = image_detect
                                mask[padding:y_count, padding:x_count] = 1
                                shape = (y_count - padding, x_count - padding)
                        # nodata = nodata.astype(bool)
                            image_detect = img_temp
                        mask = (mask != 0)
                        list_image_detect.append(cv2.resize(image_detect.astype(np.uint8), (480,480), interpolation = cv2.INTER_AREA))
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
                                    y_pred0 = (y_pred0[0,...,0] > 0.5).astype(np.uint8)
                                    mask0= list_mask[i]
                                    shape0 = list_shape[i]
                                # y_pred = np.argmax(y_pred,axis=-1)
                                # print(y_pred.shape)
                                    y_pred0 = (cv2.resize(y_pred0, (480,480), interpolation = cv2.INTER_AREA)>0.5).astype(np.uint8)
                                    y = y_pred0[mask0].reshape(shape0)

                                    
                                    r.write(y[np.newaxis,...], window=Window(start_x, start_y, shape0[1], shape0[0]))

            chunks = batch_split(list_coordinates,n=batch_size)

            for coordsss in tqdm(chunks):
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
    # img_path = r"/mnt/Nam/public/Linh/mongolia_fullmonth/LC09_L2SP_131027_20220214_20220216_02_T1.tif"
    # out_path = r"/mnt/Nam/public/Linh/mongolia_fullmonth/LC09_L2SP_131027_20220214_20220216_02_T1_crop.tif"
    # model_path = r"/mnt/Nam/tmp_Nam/pre-processing/data_train_crop_landsat/weights/u2net_crop_landsat_predict.h5"

    img_path = r"/home/skymap/data/farm_predict/image/District_0_cog.tif"
    out_path = img_path.replace(".tif", "_ressult.tif")
    model_path = r"/home/skymap/data/farm_predict/farm_predict/model_farm.h5"
    
    size = 480
    threshold_distance = 3 #ngưỡng làm mượt polygon
    threshold_connect = 3 #5 #ngưỡng nối điểm gần nhau

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', help='foo help', default=img_path)
    parser.add_argument('--output_path', help='foo help', default=out_path)
    parser.add_argument('--model_path', help='foo help', default=model_path)
    args = parser.parse_args()

    # from model import build_model
    # model = build_model((None,None,3), 32)
    # model.load_weights(model_path)
    model = tf.keras.models.load_model(model_path)
    model.trainable = False
    # cache_path = args.output_path.replace('.geojson', '.tif')
    
    predict_farm(model, args.input_path, args.output_path, size)
    # raster_to_vector(cache_path, args.output_path, threshold_distance, threshold_connect)
    # os.remove(cache_path)
    
    

