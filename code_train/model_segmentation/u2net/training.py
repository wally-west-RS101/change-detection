import tensorflow as tf
import numpy as np
from tqdm import tqdm
import glob, shutil, warnings, os

from utils.color_image import Color_image

from utils.augmentations import Augmentations
from multiprocessing import Pool
from models.import_module import DexiNed, Model_U2Netp, Model_U2Net, Adalsn, Model_UNet3plus, \
                        weighted_cross_entropy_loss, pre_process_binary_cross_entropy, IoULoss,binary_focal_loss_fixed
                        
from tensorflow.compat.v1.keras.backend import set_session
####
import tensorflow as tf
from tensorflow.keras import layers, backend, Model, utils
from matplotlib import pyplot as plt
import rasterio
import cv2
from tensorflow.compat.v1.keras.backend import set_session
warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))
num_bands = 6
input_size = 512
def binary_focal_loss_fixed(y_true, y_pred):
    gamma=2.
    alpha=.25
    y_true = tf.cast(y_true, tf.float32)
    epsilon = backend.epsilon()
    y_pred = backend.clip(y_pred, epsilon, 1.0 - epsilon)

    p_t = tf.where(backend.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_factor = backend.ones_like(y_true) * alpha

    alpha_t = tf.where(backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    cross_entropy = -backend.log(p_t)
    weight = alpha_t * backend.pow((1 - p_t), gamma)
    loss = weight * cross_entropy
    loss = backend.mean(backend.sum(loss, axis=1))
    return loss
class DataParser():
    def __init__(self, annotation, batch_size, augmentations=None, color_image=None):
        self.total_data = annotation
        self.batch_size = batch_size
        self.len_data = int(len(self.total_data)//self.batch_size)
        self.check_batch = self.len_data * self.batch_size
        self.augmentations = augmentations
        self.color_image = color_image
        self.num = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.num < self.check_batch:
            filename = self.total_data[self.num: self.num+self.batch_size]
            # with Pool() as p:
            #     p.map(self.get_batch, filename)
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
            #print('aaa',img_list)
            # with rasterio.open(img,'r') as img:
            #     im = img.read()[0:3].transpose(1,2,0)
            # with rasterio.open(img.replace('image','label'),'r') as mask:
            #     em = np.read().transpose(1,2,0)
                               
            im = np.load(img_list)[:,:,0:num_bands]
            
            
            em = np.load(img_list.replace('image', 'label'))
            if self.augmentations != None:
                im, em = self.augmentations(im, em)
            if self.color_image != None:
                im = self.color_image(im)
                
            # im = np.array(im/255., dtype=np.float32)
            # em = np.array(em/255., dtype=np.float32)
            im = np.array(im/255, dtype=np.float32)
            em = np.array(em, dtype=np.float32)
            # em = em.astype(np.float32)

            images.append(im)
            edgemaps.append(em)

        images   = np.asarray(images)
        edgemaps = np.asarray(edgemaps)
        #print(images.shape)
        #print(edgemaps.shape)
        return images, edgemaps
    
    def __len__(self):
        return self.len_data


def train_step(image_data, target):
    with tf.GradientTape() as tape:
        pred = my_model(image_data, training=True)
        
        
        #loss = weighted_cross_entropy_loss(target, pred)
        #loss = binary_focal_loss_fixed(target,pred)
        loss = pre_process_binary_cross_entropy(target,pred)
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
    #loss = weighted_cross_entropy_loss(target, pred)
    #loss = binary_focal_loss_fixed(target,pred)
    loss = pre_process_binary_cross_entropy(target,pred)
    return loss.numpy()

if __name__ == '__main__':
    VisEff = ()
    Augmen = Augmentations()
    
    bce = tf.keras.losses.BinaryCrossentropy()
    # my_model = DexiNed(480, 3)
    # my_model = Model_U2Net(480, 8)
    my_model = Model_U2Net(input_size, num_bands)
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
    traindata = DataParser(data_train, batch_size, Augmen)
    valdata = DataParser(data_val, batch_size)
    len_train = len(traindata)
    #print('len_train',len_train)
    
    len_val = len(valdata)
    #print('len_val',len_val)

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