

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import glob
import shutil
import warnings
import os
from multiprocessing import Pool
from models.import_module import DexiNed, Model_U2Netp, Model_U2Net, Adalsn, Model_UNet3plus, \
                        weighted_cross_entropy_loss, pre_process_binary_cross_entropy, IoULoss,binary_focal_loss_fixed
import imgaug as ia
from imgaug import augmenters as iaa                  
from tensorflow.compat.v1.keras.backend import set_session
####
import tensorflow as tf
from tensorflow.keras import layers, backend, Model, utils
from matplotlib import pyplot as plt
import rasterio
import cv2
from tensorflow.compat.v1.keras.backend import set_session

import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

warnings.filterwarnings("ignore")
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))

def get_data_multi(path_train):
    print("Getting data ...")
    img_dir = f'{path_train}/image/'
    mask_dir = f'{path_train}/label/'

    # create a list of image paths and mask paths in the directory
    img_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.tif')]
    mask_list = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.tif')]
    # print('kich thuoc data:', len(img_list), len(mask_list))
    print('xxxxxxxxxxxxxxx',len(img_list))
    print('xxxxxdddddddddddddd',len(mask_list))

    # read and normalize images and masks using multiple CPU cores
    with Pool(processes=8) as pool:
        img_mask_list = pool.starmap(read_img_mask, zip(img_list, mask_list))
    
    img_array_list = [x[0]for x in img_mask_list]
    mask_array_list = [x[1]for x in img_mask_list]

    X_train = np.stack(img_array_list, axis=0)
    y_train = np.stack(mask_array_list, axis=0)

    #custom for attunet
    # X_train = np.array(get_multi_scale(X_train))
    # y_train = np.array(get_multi_scale(y_train))

    print("Getting done ...")
    print('len_x_t',len(X_train))
    print('len_x_t',len(y_train))

    return X_train, y_train
def read_img_mask(img_path, mask_path):
    with rasterio.open(img_path) as src:
        img = src.read().squeeze()
  
        img = np.transpose(img, (1,2,0))
        # img = img[np.newaxis,...]

    with rasterio.open(mask_path) as src:
        mask = src.read().squeeze()
        mask[mask==255] = 1
        # print(np.unique(mask))
        mask = mask.astype(np.uint8)
        onehot = np.zeros((256,256,1))
        onehot[..., 0] = (mask==1).astype(np.uint8)
        
    return img, onehot

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


def create_seq_augment():
    """ Define a Sequential augmenters contains some action use for augment use imgaug lib
    Returns:
        Sequential augmenters object push to training for augmentation
    """
    # ia.seed(1)
    # Example batch of images.
    # The array has shape (32, 64, 64, 3) and dtype uint8.
    seq = iaa.Sometimes(0.8, iaa.SomeOf((1, 7),[
        # iaa.Fliplr(0.5),
        # Flip/mirror input images horizontally# horizontal flips
        # iaa.Flipud(0.5),
        # Flip/mirror input images vertically.
        iaa.Multiply((0.7, 1.4), per_channel=0.5),
        iaa.LinearContrast((0.7, 1.4)),
        iaa.MultiplyHueAndSaturation(mul_hue=(0.8, 1.2), mul_saturation=(0.8, 1.2)),
        #blur
        # iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.GaussianBlur(sigma=(0.0, 2.0)),
        iaa.SaltAndPepper(0.02),
        #gaus
        iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
        # Multiply all pixels in an image with a specific value, thereby making the image darker or brighter.
        # Multiply 50% of all images with a random value between 0.5 and 1.5
        # and multiply the remaining 50% channel-wise, i.e. sample one multiplier independently per channel
        iaa.Affine(
            # scale={"x": (0.8, 1.), "y": (0.8, 1.2)},
            scale={"x": (0.8, 1.5), "y": (0.8, 1.5)},
            # Scale images to a value of 80 to 120%
            # of their original size, but do this independently per axis (i.e. sample two values per image)
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            # Translate images by -10 to +10% on x- and y-axis independently
            # rotate=(-90, 90),
            # Rotate images by -90 to 90 degrees
            #             shear=(-15, 15),
            #             cval=(0, 255),
            #             mode=ia.ALL
        )
    ]))
    return seq


def agument(image,mask,augmentation=None):
    if augmentation:
        try:
            import imgaug
            # Augmentors that are safe to apply to masks
            # Some, such as Affine, have settings that make them unsafe, so always
            # test your augmentation on masks
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                            "Fliplr", "Flipud", "CropAndPad",
                            "Affine", "PiecewiseAffine"]

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return (augmenter.__class__.__name__ in MASK_AUGMENTERS)
            # Store shapes before augmentation to compare
            image_shape = image.shape
            mask_shape = mask.shape
            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            image2 = det.augment_image(image)
            # Change mask to np.uint8 because imgaug doesn't support np.bool
            mask2 = det.augment_image(mask.astype(np.uint8),
                                    hooks=imgaug.HooksImages(activator=hook))
            # Verify that shapes didn't change
            assert image2.shape == image_shape, "Augmentation shouldn't change image size"
            assert mask2.shape == mask_shape, "Augmentation shouldn't change mask size"
            # Change mask back to bool
            mask2 = mask2.astype(np.bool)
        except Exception:
            mask2 = mask
            image2 = image
    else:
        mask2 = mask
        image2 = image
    return image2, mask2.astype(np.uint8)

    
# class DataParser():
#     def __init__(self, annotation):
#         self.total_data = annotation
#         self.batch_size = 3
#         self.steps_per_epoch = int(len(self.total_data) // self.batch_size)
#         self.check_batch = self.steps_per_epoch * self.batch_size
#         self.augmentations = [self.flip_ud, self.flip_lr, self.rot90]
#         self.num = 0

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.num < self.check_batch:
#             filename = self.total_data[self.num: self.num + self.batch_size]
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
#             im, em = self.preprocess(img_list, img_list.replace('/image/', '/label/'))
#             for f in self.augmentations:
#                 if np.random.uniform() < 0.3:
#                     im, em = f(im, em)
#                 elif 0.3 < np.random.uniform() < 0.8:
#                     seq = create_seq_augment()
#                     im, em = agument(im, em, seq)
#             images.append(im)
#             edgemaps.append(em)
#         images   = np.asarray(images)
#         edgemaps = np.asarray(edgemaps)
#         return images, edgemaps

#     def preprocess(self, path_img, path_mask):
#         with rasterio.open(path_img) as img:
#             width,height = img.width, img.height
#             new_image_width = new_image_height = max(width, height)
#             values1 = img.read().transpose(1, 2, 0).astype(np.uint8)
#             image = values1 / 255.0
#         with rasterio.open(path_mask) as mas:
#             values2 = mas.read()
#             label = (values2[0] / 255.0 > 0.5).astype(np.float32)
#         return image, label[..., np.newaxis]
    
#     def flip_ud(self, im, em):
#         return np.flipud(im), np.flipud(em)

#     def flip_lr(self, im, em):
#         return np.fliplr(im), np.fliplr(em)

#     def rot90(self, im, em):
#         return np.rot90(im), np.rot90(em)
    
#     def __len__(self):
#         return self.steps_per_epoch


# def train_step(image_data, target):
#     with tf.GradientTape() as tape:
#         pred = my_model(image_data, training=True)
#         # loss = weighted_cross_entropy_loss(target, pred)
#         loss = binary_focal_loss_fixed(target, pred)
#         gradients = tape.gradient(loss, my_model.trainable_variables)
#     del tape
#     optimizer.apply_gradients(zip(gradients, my_model.trainable_variables))
#     global_steps.assign_add(1)
#     with writer.as_default():
#         tf.summary.scalar("loss/loss", loss, step=global_steps)
#     writer.flush()
        
#     return loss.numpy()


def val_step(image_data, target):
    pred = my_model(image_data, training=False)
    # loss = weighted_cross_entropy_loss(target, pred)
    loss = binary_focal_loss_fixed(target, pred)
    return loss.numpy()


if __name__ == '__main__':

    bce = tf.keras.losses.BinaryCrossentropy()
    # my_model = DexiNed(480, 3)
    # my_model = Model_U2Net(480, 8)
    my_model = Model_U2Net(256, 3)
    # my_model = Model_UNet3plus(480, 3)
    # my_model = Adalsn(480, 3)
    # my_model.load_weights(r'/home/skymap/data/Bahrain_change/train_256_bd/w/u2net_256_bd_4_b_23.h5')
    # my_model.load_weights('/mnt/data/Nam_work_space/model/u2net_farm_v3.h5')
    # path1 = glob.glob('/mnt/data/Nam_work_space/data_train/wajo_image_z11_1706_999/*.npy')
    # np.random.shuffle(path1)
    # path2 = glob.glob('/mnt/data/Nam_work_space/data_train/image_40/*.npy')
    # np.random.shuffle(path2)
    # path3 = glob.glob('/mnt/data/Nam_work_space/data_train/image_train/*.npy')
    # np.random.shuffle(path3)
    # path4 = glob.glob('/mnt/data/Nam_work_space/data_train/image_update/*.npy')
    # np.random.shuffle(path4)
    # path3 = glob.glob('/mnt/data/banana/data_train/_v10/train/image/*.npy')
    optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001)
    my_model.compile(optimizer=optimizer,
                loss=bce,
                metrics=['accuracy'])
    my_model.summary()
    # alpha = 0.8
    # data_train = 5*path1[:int(len(path1)*alpha)]+11*path2[:int(len(path2)*alpha)]+path3[:int(len(path3)*alpha)]+path4[:int(len(path4)*alpha)]
    # data_val = 5*path1[int(len(path1)*alpha):]+11*path2[int(len(path2)*alpha):]+path3[int(len(path3)*alpha):]+path4[int(len(path4)*alpha):]  
    # data_train = 3*path1[:int(len(path1)*alpha)]+9*path2[:int(len(path2)*alpha)]+path3[:int(len(path3)*alpha)]
    # data_val = 3*path1[int(len(path1)*alpha):]+9*path2[int(len(path2)*alpha):]+path3[int(len(path3)*alpha):]
    # data_train = path3[:int(len(path3)* alpha)]
    # data_val = path3[int(len(path3)* alpha):]
    # np.random.shuffle(data_train)
    # np.random.shuffle(data_val)
    from sklearn.model_selection import train_test_split
    # data_train = glob.glob('/home/skymap/data/Bahrain_change/SET14/Giaoviec_s14/data_test_bd_s14/train/image/*.tif')[0:10]
    # data_val = glob.glob('/home/skymap/data/Bahrain_change/SET14/Giaoviec_s14/data_test_bd_s14/val/image/*.tif')[0:10]
    data_train = '/home/skymap/data/Bahrain_change/SET14/Giaoviec_s14/DATA_TRAIN_Green/train'
    # data_val = '/home/skymap/data/Bahrain_change/set16/Data_s15_bd_test/val'
    # traindata = DataParser(data_train)
    # print(traindata)
    # valdata = DataParser(data_val)
    X,y = get_data_multi(data_train)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=2019)
    # valdata = get_data_multi(data_val)
    len_train = len(X_train)
    print('lllllllllll',len_train)
    len_val = len(X_valid)
    model_path = '/home/skymap/data/Bahrain_change/SET14/Giaoviec_s14/train_test_bd_s14/w/green.h5'
    # print('aaaaaa',type(traindata))
    # print('lllllllllll',len(traindata))
    # len_val = len(valdata)
    patience_early = 20
    factor =  0.1
    patience_reduce=  3
    min_lr = 0.00001
    verbose=1
    epochs = 150
    batch_size = 4

    callbacks = [
        EarlyStopping(patience=patience_early, verbose=verbose),
        ReduceLROnPlateau(factor=factor, patience=patience_reduce, min_lr=min_lr, verbose=verbose),
        ModelCheckpoint(model_path, verbose=verbose, save_best_only=True, save_weights_only=True)
    ]

    # model.fit(train_datagen.flow(X_train, y_train, batch_size=int(batch_size)), epochs=epochs, callbacks=callbacks,
    #                             validation_data=(X_valid, y_valid))
    
    my_model.fit(X_train, y_train, batch_size=int(batch_size), epochs=epochs, callbacks=callbacks,
                            validation_data=(X_valid, y_valid))
#     X_pairs = [X_train[i:i+3] for i in range(0, len(X_train), 3)]
#     Y_pairs = [y_train[i:i+3] for i in range(0, len(y_train), 3)]


#     #TRAIN_LOGDIR = '/media/skymap/Nam/tmp_Nam/pre-processing/farm_all/weight_aus/u2net'
#     TRAIN_LOGDIR = '/home/skymap/data/Bahrain_change/SET14/Giaoviec_s14/train_test_bd_s14/log'
#     TRAIN_EPOCHS = 70
#     best_val_loss = 7
#     global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     print(f'GPUs {gpus}')
#     if len(gpus) > 0:
#         try: tf.config.experimental.set_memory_growth(gpus[0], True)
#         except RuntimeError: pass
#     if os.path.exists(TRAIN_LOGDIR): 
#         try: shutil.rmtree(TRAIN_LOGDIR)
#         except: pass        
#     writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
#     validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
#     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0, clipvalue=0.5)
# #     optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
#     for epoch in range(TRAIN_EPOCHS):
#         print('Epoch: %s / %s'%(str(epoch+1), str(TRAIN_EPOCHS)))
#         with tqdm(total=len_train, desc=f'Train', postfix=dict, mininterval=0.3) as pbar:
#             total_train = 0
#             for image_data, target in zip(X_pairs, Y_pairs):
#                 results = train_step(image_data, target)
#                 total_train += results
#                 pbar.set_postfix(**{'total_loss': results})
#                 pbar.update(1)             
#             pbar.set_postfix(**{'total_train': total_train / len_train})
#             pbar.update(1)   
#         with tqdm(total=len_val, desc=f'Val', postfix=dict, mininterval=0.3) as pbar:
#             total_val = 0
#             for image_data, target in zip(X_valid, y_valid):
#                 results = val_step(image_data, target)
#                 total_val += results
#                 pbar.set_postfix(**{'total_val': results})
#                 pbar.update(1)
#             pbar.set_postfix(**{'total_val': total_val/len_val})
#             pbar.update(1)
#             with validate_writer.as_default():
#                 tf.summary.scalar("validate_loss / total_val", total_val / len_val, step=epoch)
#             validate_writer.flush()
#             if best_val_loss >= total_val / len_val:
#                 from predict_farm3_2anh import predict_farm
#                 my_model.trainable = False
#                 predict_farm(my_model, r'/home/skymap/data/Bahrain_change/train_256_bd/tesst/t1/T13_54.tif',
#                             r'/home/skymap/data/Bahrain_change/train_256_bd/tesst/t2/T13_54.tif',
#                             f'/home/skymap/data/Bahrain_change/SET14/Giaoviec_s14/train_test_bd_s14/test/out/new_model256_{epoch}.tif', cfd=0.5, num_bands=3, size=512)

#                 best_val_loss = total_val / len_val
#                 my_model.trainable = True
#                 my_model.save_weights(os.path.join("/home/skymap/data/Bahrain_change/SET14/Giaoviec_s14/train_test_bd_s14/w", f"u2net_512_S14_vaug_test_newcode_{epoch}.h5"))
