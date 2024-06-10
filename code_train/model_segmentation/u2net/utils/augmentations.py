import cv2
import numpy as np
import matplotlib.pyplot as plt
#from skimage.morphology import skeletonize

def zoom_image(image, label, prob=0.5):
    random_prob = np.random.uniform()
    if random_prob < (1 - prob):
        return image, label
    
    def preprocess_image(image, label, image_size):
        image_height, image_width = image.shape[:2]
        if image_height > image_width:
            resized_height = image_size
            resized_width = int(image_width * image_size / image_height)
        else:
            resized_height = int(image_height * image_size / image_width)
            resized_width = image_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
        # label = skeletonize(label)
        pad_h = image_size - resized_height
        pad_w = image_size - resized_width
        label = label[..., np.newaxis]
        image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')
        label = np.pad(label, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')
        return image, label
        
    def _zoom(image, label, image_size):
        label = label[...,0]
        scale = np.random.choice(np.arange(1, 1.3, 0.05))
        image_new_size= int(round(image_size * scale))
        image = cv2.resize(image, (image_new_size, image_new_size), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (image_new_size, image_new_size), interpolation=cv2.INTER_CUBIC)
        
        random_x1 = np.random.randint(0, max(image_new_size // 8, 1))
        random_y1 = np.random.randint(0, max(image_new_size // 8, 1))
        random_x2 = np.random.randint(7 * image_new_size // 8, image_new_size - 1)
        random_y2 = np.random.randint(7 * image_new_size // 8, image_new_size - 1)
        image = image[random_y1:random_y2, random_x1:random_x2]
        label = label[random_y1:random_y2, random_x1:random_x2]
        return image, label

    image_size = image.shape[1]
    image, label = _zoom(image, label, image_size)
    image, label = preprocess_image(image, label, image_size)
    return image, label

def rotate(image, label):
    def flip_ud(im, em):
        return np.flipud(im), np.flipud(em)

    def flip_lr(im, em):
        return np.fliplr(im), np.fliplr(em)

    def rot90(im, em):
        return np.rot90(im), np.rot90(em)
    
    random_id = np.random.randint(0, 3)
    if random_id == 0:
        image, label = flip_ud(image,label)
    elif random_id == 1:
        image, label = flip_lr(image,label)
    elif random_id == 2:
        image, label = rot90(image,label)
        
    return image, label

class Augmentations:
    def __init__(self, prob=1):
        self.prob = prob

    def __call__(self, image, label):
        image, label = rotate(image, label)
        # image, label = zoom_image(image, label, prob=self.prob)
        return image, label


if __name__ == '__main__':
    pass
#     img_list = '/mnt/data/Nam_work_space/data_train/image_40/box_021.npy'
#     image = np.load(img_list)[...,:3]
#     label = np.load(img_list.replace('image', 'label'))
#     im, em = Augmentations()(image, label)
    
#     plt.figure(figsize=(15, 15))
#     display_list = [im, em, image, label]
#     title = ['stard_image', 'stard_label', 'image', 'label']
#     for i in range(4):
#         plt.subplot(2, 2, i+1)
#         plt.title(title[i])
#         plt.imshow(display_list[i])
#         plt.axis('off')
#     plt.show()


    # IMG_SIZE = 256
    # import tensorflow as tf
    # label = tf.image.resize(label, [IMG_SIZE, IMG_SIZE])
    # plt.imshow(label)
    # plt.show()
    
    # import imgaug
    # from imgaug.augmentables.segmaps import SegmentationMapsOnImage
    # import imgaug.augmenters as iaa
    
    # # segmap = SegmentationMapsOnImage.resize(label, sizes=256)
    # segmap = iaa.Resize(256)(label)
    
