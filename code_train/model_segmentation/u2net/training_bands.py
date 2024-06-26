import tensorflow as tf
import numpy as np
from tqdm import tqdm
import glob, shutil, warnings, os
from utils.color_image import Color_image
from utils.augmentations import Augmentations
from multiprocessing import Pool
from models.import_module import DexiNed, Model_U2Netp, Model_U2Net, Adalsn, Model_UNet3plus, \
                        weighted_cross_entropy_loss, pre_process_binary_cross_entropy, IoULoss
                        
from tensorflow.compat.v1.keras.backend import set_session
import rasterio

warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
num_bands = 8
size =512
# class DataParser():
#     def __init__(self, annotation, batch_size, augmentations=None, color_image=None):
#         self.total_data = annotation
#         self.batch_size = batch_size
#         self.len_data = int(len(self.total_data)//self.batch_size)
#         self.check_batch = self.len_data * self.batch_size
#         self.augmentations = augmentations
#         self.color_image = color_image
#         self.num = 0

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.num < self.check_batch:
#             filename = self.total_data[self.num: self.num+self.batch_size]
#             # with Pool() as p:
#             #     p.map(self.get_batch, filename)
#             image, label = self.get_batch(filename)
#             self.num += self.batch_size
#             return image, label
#         else:
#             self.num = 0
#             np.random.shuffle(self.total_data)
#             raise StopIteration


#     def get_batch(self, batch):
#         images = []
#         edgemaps = []
#         for img_list in batch:
#             im = np.load(img_list)[...,:8]
#             em = np.load(img_list.replace('image', 'label'))
#             if self.augmentations != None:
#                 im, em = self.augmentations(im, em)
#             if self.color_image != None:
#                 im = self.color_image(im)
                
#             im = np.array(im/255., dtype=np.float32)
#             em = np.array(em/255., dtype=np.float32)
#             # em = em.astype(np.float32)

#             images.append(im)
#             edgemaps.append(em)

#         images   = np.asarray(images)
#         edgemaps = np.asarray(edgemaps)

#         return images, edgemaps
    
#     def __len__(self):
#         return self.len_data

class DataParser():
    def __init__(self, annotation):
        self.total_data = annotation
        self.batch_size = 2
        self.steps_per_epoch = int(len(self.total_data)//self.batch_size)
        self.check_batch = self.steps_per_epoch * self.batch_size
        self.augmentations = [self.flip_ud, self.flip_lr, self.rot90]
        self.num = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.num < self.check_batch:
            filename = self.total_data[self.num: self.num+self.batch_size]

            image, label = self.get_batch(filename)
            self.num += self.batch_size
            return image, label
        else:
            self.num = 0
            np.random.shuffle(self.total_data)
            raise StopIteration


    def get_batch(self, batch):
        images = []
        edgemaps = []
        for img_list in batch:
            im, em = self.preprocess(img_list, img_list.replace('image', 'label'))

            for f in self.augmentations:
                if np.random.uniform()<0.20:
                    im, em=f(im, em)

            images.append(im)
            edgemaps.append(em)

        images   = np.asarray(images)
        edgemaps = np.asarray(edgemaps)

        return images, edgemaps

    def preprocess(self, path_img, path_mask):
        with rasterio.open(path_img) as img:
            width,height = img.width,img.height
            new_image_width = new_image_height = max(width,height)
            values1 = img.read().transpose(1,2,0).astype(np.uint8)
            
            #print('img', values1.shape)
#             x_center = (new_image_width - width) // 2
#             y_center = (new_image_height - height) // 2
#             result = np.full((new_image_height,new_image_width, 3), (0,0,0), dtype=np.uint8)
#             result[y_center:y_center+height, x_center:x_center+width] = values
#             result = cv2.resize(result,(320,320), interpolation = cv2.INTER_CUBIC)
            image = values1/255.0
        with rasterio.open(path_mask) as mas:
            values2 = mas.read()
            print(values2.shape)
#             result = np.full((new_image_height,new_image_width), 0, dtype=np.uint8)
#             result[y_center:y_center+height, x_center:x_center+width] = values[0]
#             result = cv2.resize(result,(320,320), interpolation = cv2.INTER_CUBIC)
            label = values2/255
            label = (values2[0] > 0.5).astype(np.float32)
            print('img', label.shape)
        return image, label[...,np.newaxis]
    
    def flip_ud(self, im, em):
        return np.flipud(im), np.flipud(em)

    def flip_lr(self, im, em):
        return np.fliplr(im), np.fliplr(em)

    def rot90(self, im, em):
        return np.rot90(im), np.rot90(em)
    
    def __len__(self):
        return self.steps_per_epoch

def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred = my_model(image_data, training=True)
        loss = weighted_cross_entropy_loss(target, pred)
        gradients = tape.gradient(loss, my_model.trainable_variables)
    del tape
    optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))
    global_steps.assign_add(1)
    with writer.as_default():
        tf.summary.scalar("loss/loss", loss, step=global_steps)
    writer.flush()
        
    return loss.numpy()

def val_step(image_data, target):
    pred = my_model(image_data, training=False)
    # print('pred',pred)
    loss = weighted_cross_entropy_loss(target, pred)
    # print('loss', loss)
    return loss.numpy()

if __name__ == '__main__':
#     VisEff = Color_image()
#     Augmen = Augmentations()
    
    bce = tf.keras.losses.BinaryCrossentropy()
    # my_model = DexiNed(480, 3)
    # my_model = Model_U2Net(480, 8)
    my_model = Model_U2Net(size, num_bands)
    # my_model = Model_UNet3plus(480, 3)
    # my_model = Adalsn(480, 3)
    #my_model.load_weights(r'')
    # my_model.load_weights('/mnt/data/Nam_work_space/model/u2net_farm_v3.h5')

    # path1 = glob.glob('/mnt/data/Nam_work_space/data_train/wajo_image_z11_1706_999/*.npy')
    # np.random.shuffle(path1)
    # path2 = glob.glob('/mnt/data/Nam_work_space/data_train/image_40/*.npy')
    # np.random.shuffle(path2)
    # path3 = glob.glob('/mnt/data/Nam_work_space/data_train/image_train/*.npy')
    # np.random.shuffle(path3)
    # path4 = glob.glob('/mnt/data/Nam_work_space/data_train/image_update/*.npy')
    # np.random.shuffle(path4)

    # # path3 = glob.glob('/mnt/data/banana/data_train/_v10/train/image/*.npy')
    # # np.random.shuffle(path3)
    
    # alpha = 0.8
    # data_train = 5*path1[:int(len(path1)*alpha)]+11*path2[:int(len(path2)*alpha)]+path3[:int(len(path3)*alpha)]+path4[:int(len(path4)*alpha)]
    # data_val = 5*path1[int(len(path1)*alpha):]+11*path2[int(len(path2)*alpha):]+path3[int(len(path3)*alpha):]+path4[int(len(path4)*alpha):]  
    # # data_train = 3*path1[:int(len(path1)*alpha)]+9*path2[:int(len(path2)*alpha)]+path3[:int(len(path3)*alpha)]
    # # data_val = 3*path1[int(len(path1)*alpha):]+9*path2[int(len(path2)*alpha):]+path3[int(len(path3)*alpha):]
    # # data_train = path3[:int(len(path3)* alpha)]
    # # data_val = path3[int(len(path3)* alpha):]
    
    # np.random.shuffle(data_train)
    # np.random.shuffle(data_val)
    
    data_train = glob.glob(r'/home/skymap/big_data/Dao_work_space/datachange_full/data_train_other/data1train/image/*.npy')
    data_val = glob.glob(r'/home/skymap/big_data/Dao_work_space/datachange_full/data_train_other/data1val/image/*.npy')
    
    batch_size = 2
    traindata = DataParser(data_train)
    valdata = DataParser(data_val)
    len_train = len(traindata)
    len_val = len(valdata)

    #TRAIN_LOGDIR = '/media/skymap/Nam/tmp_Nam/pre-processing/farm_all/weight_aus/u2net'
    TRAIN_LOGDIR = '/home/skymap/big_data/Dao_work_space/datachange_full/data_train_other/train'
    TRAIN_EPOCHS = 50
    best_val_loss = 10
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)


    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPUs {gpus}')
    if len(gpus) > 0:
        try: tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError: pass
    if os.path.exists(TRAIN_LOGDIR): 
        try: shutil.rmtree(TRAIN_LOGDIR)
        except: pass
        
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for epoch in range(TRAIN_EPOCHS):
        print('Epoch: %s / %s'%(str(epoch+1),str(TRAIN_EPOCHS)))
        with tqdm(total=len_train,desc=f'Train',postfix=dict,mininterval=0.3) as pbar:
            total_train = 0
            for image_data, target in traindata:
                results = train_step(image_data, target)
                total_train += results
                pbar.set_postfix(**{'total_loss' : results})
                pbar.update(1)              
            pbar.set_postfix(**{'total_train' : total_train/len_train})
            pbar.update(1)    
                 
        with tqdm(total=len_val,desc=f'Val',postfix=dict,mininterval=0.3) as pbar:
            total_val = 0
            for image_data, target in valdata:
                results = val_step(image_data, target)
                total_val += results
                pbar.set_postfix(**{'total_val' : results})
                pbar.update(1)
            pbar.set_postfix(**{'total_val' : total_val/len_val})
            pbar.update(1)
            with validate_writer.as_default():
                tf.summary.scalar("validate_loss/total_val", total_val/len_val, step=epoch)
            validate_writer.flush()
            
            if best_val_loss>=total_val/len_val:
                my_model.save_weights(os.path.join("/home/skymap/big_data/Dao_work_space/datachange_full/data_train_other/weights", f"u2net_other_v2.h5"))
                best_val_loss = total_val/len_val

        print(22*'-'+7*'*'+22*'-')
        print() 