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
from models.import_module import DexiNed, Model_U2Netp, Model_U2Net, Adalsn, Model_UNet3plus, \
                        weighted_cross_entropy_loss, pre_process_binary_cross_entropy, IoULoss
from tensorflow.keras import layers, backend, Model, utils

warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))
from tensorflow.keras import layers, Model, Input
import tensorflow as tf

class U2Net:
    def __init__(self,input_size):
        self.input_size = input_size
        self.model = self.Model_U2Net(self.input_size)
        self.model.load_weights(r'/home/skymap/big_data/Dao_work_space/OpenLandstraindata/code_train/weights_waterbody_u2net/u2net.h5')

    def _upsample_like(self,src,tar):
        # src = tf.image.resize(images=src, size=tar.shape[1:3], method= 'bilinear')
        h = int(tar.shape[1]/src.shape[1])
        w = int(tar.shape[2]/src.shape[2])
        src = layers.UpSampling2D((h,w),interpolation='bilinear')(src)
        return src

    def REBNCONV(self,x,out_ch=3,dirate=1):
        # x = layers.ZeroPadding2D(1*dirate)(x)
        x = layers.Conv2D(out_ch, 3, padding = "same", dilation_rate=1*dirate)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def RSU7(self,hx, mid_ch=12, out_ch=3):
        hxin = self.REBNCONV(hx, out_ch,dirate=1)

        hx1 = self.REBNCONV(hxin, mid_ch,dirate=1)
        hx = layers.MaxPool2D(2,strides=2)(hx1)

        hx2 = self.REBNCONV(hx, mid_ch,dirate=1)
        hx = layers.MaxPool2D(2,strides=2)(hx2)

        hx3 = self.REBNCONV(hx, mid_ch,dirate=1)
        hx = layers.MaxPool2D(2,strides=2)(hx3)

        hx4 = self.REBNCONV(hx, mid_ch,dirate=1)
        hx = layers.MaxPool2D(2,strides=2)(hx4)

        hx5 = self.REBNCONV(hx, mid_ch,dirate=1)
        hx = layers.MaxPool2D(2,strides=2)(hx5)

        hx6 = self.REBNCONV(hx, mid_ch,dirate=1)

        hx7 = self.REBNCONV(hx6, mid_ch,dirate=2)

        hx6d = self.REBNCONV(layers.concatenate([hx7,hx6]), mid_ch,dirate=1)
        hx6dup = self._upsample_like(hx6d,hx5)

        hx5d = self.REBNCONV(layers.concatenate([hx6dup,hx5]), mid_ch,dirate=1)
        hx5dup = self._upsample_like(hx5d,hx4)

        hx4d = self.REBNCONV(layers.concatenate([hx5dup,hx4]), mid_ch,dirate=1)
        hx4dup = self._upsample_like(hx4d,hx3)

        hx3d = self.REBNCONV(layers.concatenate([hx4dup,hx3]), mid_ch,dirate=1)
        hx3dup = self._upsample_like(hx3d,hx2)

        hx2d = self.REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
        hx2dup = self._upsample_like(hx2d,hx1)

        hx1d = self.REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

        return hx1d + hxin

    def RSU6(self,hx, mid_ch=12, out_ch=3):
        hxin = self.REBNCONV(hx, out_ch,dirate=1)

        hx1 = self.REBNCONV(hxin, mid_ch,dirate=1)
        hx = layers.MaxPool2D(2,strides=2)(hx1)

        hx2 = self.REBNCONV(hx, mid_ch,dirate=1)
        hx = layers.MaxPool2D(2,strides=2)(hx2)

        hx3 = self.REBNCONV(hx, mid_ch,dirate=1)
        hx = layers.MaxPool2D(2,strides=2)(hx3)

        hx4 = self.REBNCONV(hx, mid_ch,dirate=1)
        hx = layers.MaxPool2D(2,strides=2)(hx4)

        hx5 = self.REBNCONV(hx, mid_ch,dirate=1)

        hx6 = self.REBNCONV(hx, mid_ch,dirate=2)


        hx5d =  self.REBNCONV(layers.concatenate([hx6,hx5]), mid_ch,dirate=1)
        hx5dup = self._upsample_like(hx5d,hx4)

        hx4d = self.REBNCONV(layers.concatenate([hx5dup,hx4]), mid_ch,dirate=1)
        hx4dup = self._upsample_like(hx4d,hx3)

        hx3d = self.REBNCONV(layers.concatenate([hx4dup,hx3]), mid_ch,dirate=1)
        hx3dup = self._upsample_like(hx3d,hx2)

        hx2d = self.REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
        hx2dup = self._upsample_like(hx2d,hx1)

        hx1d = self.REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

        return hx1d + hxin

    def RSU5(self,hx, mid_ch=12, out_ch=3):
        hxin = self.REBNCONV(hx, out_ch,dirate=1)

        hx1 = self.REBNCONV(hxin, mid_ch,dirate=1)
        hx = layers.MaxPool2D(2,strides=2)(hx1)

        hx2 = self.REBNCONV(hx, mid_ch,dirate=1)
        hx = layers.MaxPool2D(2,strides=2)(hx2)

        hx3 = self.REBNCONV(hx, mid_ch,dirate=1)
        hx = layers.MaxPool2D(2,strides=2)(hx3)

        hx4 = self.REBNCONV(hx, mid_ch,dirate=1)

        hx5 = self.REBNCONV(hx4, mid_ch,dirate=2)

        hx4d = self.REBNCONV(layers.concatenate([hx5,hx4]), mid_ch,dirate=1)
        hx4dup = self._upsample_like(hx4d,hx3)

        hx3d = self.REBNCONV(layers.concatenate([hx4dup,hx3]), mid_ch,dirate=1)
        hx3dup = self._upsample_like(hx3d,hx2)

        hx2d = self.REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
        hx2dup = self._upsample_like(hx2d,hx1)

        hx1d = self.REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

        return hx1d + hxin


    def RSU4(self,hx,mid_ch=12, out_ch=3):
        hxin = self.REBNCONV(hx, out_ch,dirate=1)

        hx1 = self.REBNCONV(hxin,mid_ch,dirate=1)
        hx = layers.MaxPool2D(2,strides=2)(hx1)

        hx2 = self.REBNCONV(hx, mid_ch,dirate=1)
        hx = layers.MaxPool2D(2,strides=2)(hx2)

        hx3 = self.REBNCONV(hx, mid_ch,dirate=1)

        hx4 = self.REBNCONV(hx3, mid_ch,dirate=2)
        hx3d = self.REBNCONV(layers.concatenate([hx4,hx3]), mid_ch,dirate=1)
        hx3dup = self._upsample_like(hx3d,hx2)

        hx2d = self.REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
        hx2dup = self._upsample_like(hx2d,hx1)

        hx1d = self.REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

        return hx1d + hxin

    def RSU4F(self,hx, mid_ch=12, out_ch=3):
        hxin = self.REBNCONV(hx,out_ch,dirate=1)

        hx1 = self.REBNCONV(hxin, mid_ch,dirate=1)
        hx2 = self.REBNCONV(hx1, mid_ch,dirate=2)
        hx3 = self.REBNCONV(hx2, mid_ch,dirate=4)

        hx4 = self.REBNCONV(hx3, mid_ch,dirate=8)

        hx3d = self.REBNCONV(layers.concatenate([hx4,hx3]), mid_ch,dirate=4)
        hx2d = self.REBNCONV(layers.concatenate([hx3d,hx2]), mid_ch,dirate=2)
        hx1d = self.REBNCONV(layers.concatenate([hx2d,hx1]), out_ch,dirate=1)

        return hx1d + hxin

    def U2NET(self,hx, out_ch=1):
        # hx = Input(shape=(480,480,3))
        #stage 1
        hx1 = self.RSU7(hx, 32,64)
        hx = layers.MaxPool2D(2,strides=2)(hx1)

        #stage 2
        hx2 = self.RSU6(hx, 32,128)
        hx = layers.MaxPool2D(2,strides=2)(hx2)

        #stage 3
        hx3 = self.RSU5(hx, 64,256)
        hx = layers.MaxPool2D(2,strides=2)(hx3)

        #stage 4
        hx4 = self.RSU4(hx, 128,512)
        hx = layers.MaxPool2D(2,strides=2)(hx4)

        #stage 5
        hx5 = self.RSU4F(hx, 256,512)
        hx = layers.MaxPool2D(2,strides=2)(hx5)

        #stage 6
        hx6 = self.RSU4F(hx, 256,512)
        hx6up = self._upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.RSU4F(layers.concatenate([hx6up,hx5]), 256,512)
        hx5dup = self._upsample_like(hx5d,hx4)

        hx4d = self.RSU4(layers.concatenate([hx5dup,hx4]), 128,256)
        hx4dup = self._upsample_like(hx4d,hx3)

        hx3d = self.RSU5(layers.concatenate([hx4dup,hx3]), 64,128)
        hx3dup = self._upsample_like(hx3d,hx2)

        hx2d = self.RSU6(layers.concatenate([hx3dup,hx2]), 32,64)
        hx2dup = self._upsample_like(hx2d,hx1)

        hx1d = self.RSU7(layers.concatenate([hx2dup,hx1]), 16,64)


        #side output
        d1 = layers.Conv2D(1, 3,padding="same")(hx1d)

        d2 = layers.Conv2D(1, 3,padding="same")(hx2d)
        d2 = self._upsample_like(d2,d1)

        d3 = layers.Conv2D(1, 3,padding="same")(hx3d)
        d3 = self._upsample_like(d3,d1)

        d4 = layers.Conv2D(1, 3,padding="same")(hx4d)
        d4 = self._upsample_like(d4,d1)

        d5 = layers.Conv2D(1, 3,padding="same")(hx5d)
        d5 = self._upsample_like(d5,d1)

        d6 = layers.Conv2D(1, 3,padding="same")(hx6)
        d6 = self._upsample_like(d6,d1)

        d0 = layers.Conv2D(out_ch,1)(layers.concatenate([d1,d2,d3,d4,d5,d6]))

        o1    = layers.Activation('sigmoid')(d1)
        o2    = layers.Activation('sigmoid')(d2)
        o3    = layers.Activation('sigmoid')(d3)
        o4    = layers.Activation('sigmoid')(d4)
        o5    = layers.Activation('sigmoid')(d5)
        o6    = layers.Activation('sigmoid')(d6)
        ofuse = layers.Activation('sigmoid')(d0)

        return [ofuse, o1, o2, o3, o4, o5, o6]

    def U2NETP(self,hx, out_ch=1):
        # hx = Input(shape=(480,480,3))
        #stage 1
        hx1 = self.RSU7(hx, 16,64)
        hx = layers.MaxPool2D(2,strides=2)(hx1)

        #stage 2
        hx2 = self.RSU6(hx, 16,64)
        hx = layers.MaxPool2D(2,strides=2)(hx2)

        #stage 3
        hx3 = self.RSU5(hx, 16,64)
        hx = layers.MaxPool2D(2,strides=2)(hx3)

        #stage 4
        hx4 = self.RSU4(hx, 16,64)
        hx = layers.MaxPool2D(2,strides=2)(hx4)

        #stage 5
        hx5 = self.RSU4F(hx, 16,64)
        hx = layers.MaxPool2D(2,strides=2)(hx5)

        #stage 6
        hx6 = self.RSU4F(hx, 16,64)
        hx6up = self._upsample_like(hx6,hx5)

        #-------------------- decoder --------------------
        hx5d = self.RSU4F(layers.concatenate([hx6up,hx5]), 16,64)
        hx5dup = self._upsample_like(hx5d,hx4)

        hx4d = self.RSU4(layers.concatenate([hx5dup,hx4]), 16,64)
        hx4dup = self._upsample_like(hx4d,hx3)

        hx3d = self.RSU5(layers.concatenate([hx4dup,hx3]), 16,64)
        hx3dup = self._upsample_like(hx3d,hx2)

        hx2d = self.RSU6(layers.concatenate([hx3dup,hx2]), 16,64)
        hx2dup = self._upsample_like(hx2d,hx1)

        hx1d = self.RSU7(layers.concatenate([hx2dup,hx1]), 16,64)


        #side output
        d1 = layers.Conv2D(1, 3,padding="same")(hx1d)

        d2 = layers.Conv2D(1, 3,padding="same")(hx2d)
        d2 = self._upsample_like(d2,d1)

        d3 = layers.Conv2D(1, 3,padding="same")(hx3d)
        d3 = self._upsample_like(d3,d1)

        d4 = layers.Conv2D(1, 3,padding="same")(hx4d)
        d4 = self._upsample_like(d4,d1)

        d5 = layers.Conv2D(1, 3,padding="same")(hx5d)
        d5 = self._upsample_like(d5,d1)

        d6 = layers.Conv2D(1, 3,padding="same")(hx6)
        d6 = self._upsample_like(d6,d1)

        d0 = layers.Conv2D(out_ch,1)(layers.concatenate([d1,d2,d3,d4,d5,d6]))

        o1    = layers.Activation('sigmoid')(d1)
        o2    = layers.Activation('sigmoid')(d2)
        o3    = layers.Activation('sigmoid')(d3)
        o4    = layers.Activation('sigmoid')(d4)
        o5    = layers.Activation('sigmoid')(d5)
        o6    = layers.Activation('sigmoid')(d6)
        ofuse = layers.Activation('sigmoid')(d0)

        return tf.stack([ofuse, o1, o2, o3, o4, o5, o6])

    def Model_U2Net(self,input_size):
        hx = Input(shape=(input_size,input_size,3))
        out = self.U2NET(hx)
        model = Model(inputs = hx, outputs = out)
        return model
    def predict(self,data):
        result = self.model.predict(data)
        return result

#     def Model_U2Netp(image_size, num_band):
#         hx = Input(shape=(image_size,image_size,num_band))
#         out = U2NETP(hx)
#         model = Model(inputs = hx, outputs = out)
#         return model

#     if __name__ == '__main__':
#         pass

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
#     with rasterio.open(img) as r:
#         num_band = r.count
#         for chanel in range(1,num_band+1):
#             data = r.read(chanel).astype(np.float16)
#             data[data==0] = np.nan
#             qt_scheme.append({
#                 'p2': np.nanpercentile(data, 2),
#                 'p98': np.nanpercentile(data, 98),
#             })
#     # print(qt_scheme)
    return qt_scheme
# def get_quantile_schema(img):
#     qt_scheme = []
#     with COGReader(img) as cog:
#         stats = cog.stats()
#         for _, value in stats.items():
#             qt_scheme.append({
#                 'p2': value['percentile_2'],
#                 'p98': value['percentile_98'],
#             })
#     return qt_scheme
    
def predict_farm(model, path_image, path_predict, size=480):
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
                    values = raster.read(window=read_wd)[0:3]
                if raster.profile["dtype"]=="uint8":
                    image_detect = values.transpose(1,2,0).astype(int)
                else:
                    datas = []
                    for chain_i in range(8):
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
                mask = (mask!=0)
                    
                if np.count_nonzero(image_detect) > 0:
                    if len(np.unique(image_detect)) <= 2:
                        pass
                    else:
                        y_pred = model.predict(image_detect[np.newaxis,...]/255.)[0]
                        y_pred = (y_pred[0,...,0] > 0.9).astype(np.uint8)
                        y = y_pred[mask].reshape(shape)

                        with write_lock:
                            r.write(y[np.newaxis,...], window=Window(start_x, start_y, shape[1], shape[0]))
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
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
    
    Vectorization.save_polygon(np.pad(skeleton, pad_width=1).astype(np.intc), threshold_distance, threshold_connect, transform, projstr, path_out)
    print("Done!!!")

if __name__=="__main__":
    img_path = r"/home/skymap/big_data/Dao_work_space/OpenLandstraindata/imag1e/20220814_103841_ssc10_u0002_visual.tif"
    out_path = r"/home/skymap/big_data/Dao_work_space/OpenLandstraindata/imag1e/imag1e_water/2.tif"
    model_path = r"/home/skymap/big_data/Dao_work_space/OpenLandstraindata/code_train/weights_waterbody_u2net/u2net.h5"
    
    size = 480
    threshold_distance = 3 #ngưỡng làm mượt polygon
    threshold_connect = 5 #ngưỡng nối điểm gần nhau

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_path', help='foo help', default=img_path)
    # parser.add_argument('--output_path', help='foo help', default=out_path)
    # parser.add_argument('--model_path', help='foo help', default=model_path)
    # args = parser.parse_args()
    
    # model_farm = tf.keras.models.load_model(args.model_path)
    model = U2Net(size)
    
    #model_farm = tf.keras.models.load_model(model_path)
    
    # cache_path = args.output_path.replace('.geojson', '.tif')
    
    predict_farm(model, img_path, out_path, size)
    # raster_to_vector(cache_path, args.output_path, threshold_distance, threshold_connect)
    # os.remove(cache_path)
    
    

