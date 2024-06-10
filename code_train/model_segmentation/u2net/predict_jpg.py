from PIL import Image
import numpy as np
import concurrent.futures
from tqdm import tqdm
import threading
import glob, os

import tensorflow as tf
# import Vectorization
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects
# from rio_tiler.io import COGReader
from tensorflow.compat.v1.keras.backend import set_session
import os, glob

# warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))
num_bands = 3
size = 256
def predict_farm(model, path_image, path_predict, size=512):

    
    img = Image.open(path_image)
    width, height = img.size
    print(img.size)
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

    num_bands = 3 
    image_data = np.array(img)

    with Image.new('L', (width, height)) as result_img:
        result_data = np.array(result_img)

        read_lock = threading.Lock()
        write_lock = threading.Lock()

        def process(coordinates):
            x_off, y_off, x_count, y_count, start_x, start_y = coordinates
            read_wd = (x_off, y_off, x_off + x_count, y_off + y_count)
            with read_lock:
                values = image_data[y_off:y_off + y_count, x_off:x_off + x_count, :num_bands]

            if image_data.dtype == 'uint8':
                image_detect = values.astype(int)
        

            img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
            mask = np.pad(np.ones((stride_size, stride_size), dtype=np.uint8), ((padding, padding), (padding, padding)))
            shape = (stride_size, stride_size)

            if y_count < input_size or x_count < input_size:
                img_temp = np.zeros((input_size, input_size, image_detect.shape[2]))
                mask = np.zeros((input_size, input_size), dtype=np.uint8)

                if start_x == 0 and start_y == 0:
                    img_temp[(input_size - y_count):input_size, (input_size - x_count):input_size] = image_detect
                    mask[(input_size - y_count):input_size - padding, (input_size - x_count):input_size - padding] = 1
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

                image_detect = img_temp

            mask = (mask != 0)

            if np.count_nonzero(image_detect) > 0:
                if len(np.unique(image_detect)) <= 2:
                    pass
                else:
                    y_pred = model.predict(image_detect[np.newaxis, ...] / 255.)

                    # Chuyển đổi thành numpy array
                    y_pred = np.array(y_pred)

                    # Áp dreeshold
                    y_pred = (y_pred[0, 0, ..., 0] > 0.5).astype(np.uint8)

                    y = y_pred[mask].reshape(shape)
         

                    with write_lock:
                        result_data[start_y:start_y + shape[0], start_x:start_x + shape[1]] = y

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            results = list(tqdm(executor.map(process, list_coordinates), total=len(list_coordinates)))

        result_img.putdata(result_data.flatten())
        result_img.save(path_predict)

        return path_predict
    
if __name__=="__main__":
    model_path = r'/home/skymap/data/CHUYENDOISOVT/DATA_train/train_v1/w/u2net_256_edge_buff_add_aug_v1_model.h5'
    # dir_img = r"/home/skymap/data/Bahrain_change/set999/stack_4b"
    # dir_out = r"/home/skymap/data/Bahrain_change/set999/out_set_9/vege"
    #dir_name = r"/home/skymap/big_data/Dao_work_space/OpenLandstraindata/code/name.txt"
    dir_img = '/home/skymap/data/CHUYENDOISOVT/test_module/test_tif/2.PNG'
    dir_out = '/home/skymap/data/CHUYENDOISOVT/test_module/test_tif/out'
    os.makedirs(dir_out, exist_ok=True)
    list_img = glob.glob(os.path.join(dir_img,'*.'))
    print(len(list_img), 'all')
    
    count = 0
    
    
    model_farm = tf.keras.models.load_model(model_path)
    # model_farm1 = tf.keras.models.load_model(model_path)
    # model_farm2 = tf.keras.models.load_model(model_path)
    # model_farm3 = tf.keras.models.load_model(model_path)
    size = 256
    # name = []
    # with open(dir_name,'r') as f:
    #     name = [line.rstrip('\n') for line in f]

    # for input_path in list_img:
    #     if (os.path.basename(input_path) in name):
    #         print(f'xu li anh {os.path.basename(input_path)}')
    #         output_path = os.path.join(dir_out, os.path.basename(input_path))
    # for input_path in list_img:
    #     print('count',count)
    output_path = os.path.join(dir_out,os.path.basename(dir_img))
    predict_farm(model_farm, dir_img, output_path, size)
        # count+=1
            
                

