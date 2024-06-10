


import numpy as np

import glob, shutil, warnings, os


import imgaug as ia
from imgaug import augmenters as iaa                  


from matplotlib import pyplot as plt
import rasterio
import cv2

import matplotlib.pyplot as plt
def create_seq_augment():
    """ Define a Sequential augmenters contains some action use for augment use imgaug lib
    Returns:
        Sequential augmenters object push to training for augmentation
    """
    # ia.seed(1)
    # Example batch of images.
    # The array has shape (32, 64, 64, 3) and dtype uint8.
    seq = iaa.Sometimes(1, iaa.SomeOf((1,1),[
        iaa.Fliplr(0.5),
        # Flip/mirror input images horizontally# horizontal flips
        # iaa.Flipud(0.5),
        # Flip/mirror input images vertically.
        # iaa.Multiply((0.6, 1.5), per_channel=0.5),
        #blur
        # iaa.GaussianBlur(sigma=(0.0, 3.0)),
        #gaus
        # iaa.AdditiveGaussianNoise(scale=(0, 0.1*255)),
        # Multiply all pixels in an image with a specific value, thereby making the image darker or brighter.
        # Multiply 50% of all images with a random value between 0.5 and 1.5
        # and multiply the remaining 50% channel-wise, i.e. sample one multiplier independently per channel
        # iaa.Affine(
        #     # scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        #     # Scale images to a value of 80 to 120%
        #     # of their original size, but do this independently per axis (i.e. sample two values per image)
        #     # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        #     # Translate images by -10 to +10% on x- and y-axis independently
        #     rotate=(-90, 90),
        #     # Rotate images by -90 to 90 degrees
        #     #             shear=(-15, 15),
        #     #             cval=(0, 255),
        #     #             mode=ia.ALL
        # )
    ]))
    return seq

# def agument(image,mask,augmentation=None):
#     if augmentation:
#         try:
#             import imgaug

#             # Augmentors that are safe to apply to masks
#             # Some, such as Affine, have settings that make them unsafe, so always
#             # test your augmentation on masks
#             MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
#                             "Fliplr", "Flipud", "CropAndPad",
#                             "Affine", "PiecewiseAffine"]

#             def hook(images, augmenter, parents, default):
#                 """Determines which augmenters to apply to masks."""
#                 return (augmenter.__class__.__name__ in MASK_AUGMENTERS)

#             # Store shapes before augmentation to compare
#             image_shape = image.shape
#             mask_shape = mask.shape
#             # Make augmenters deterministic to apply similarly to images and masks
#             det = augmentation.to_deterministic()
#             image2 = det.augment_image(image)
#             # Change mask to np.uint8 because imgaug doesn't support np.bool
#             mask2 = det.augment_image(mask.astype(np.uint8),
#                                     hooks=imgaug.HooksImages(activator=hook))
#             # Verify that shapes didn't change
#             assert image2.shape == image_shape, "Augmentation shouldn't change image size"
#             assert mask2.shape == mask_shape, "Augmentation shouldn't change mask size"
#             # Change mask back to bool
#             mask2 = mask2.astype(np.bool)
#         except Exception:
#             mask2 = mask
#             image2 = image
#     else:
#         mask2 = mask
#         image2 = image
#     return image2, mask2

# path_img = '/home/skymap/data/CHUYENDOISOVT/DATA_train/DATA_TRAIN_BF_V2/val/image/0001_mask_box_2_10.tif'
# path_mask = '/home/skymap/data/CHUYENDOISOVT/DATA_train/DATA_TRAIN_BF_V2/val/label/0001_mask_box_2_10.tif'
# for i in range(10):
#     with rasterio.open(path_img,'r') as src:
#         image = src.read()
    
#         image1 = image.transpose(1,2,0)
#         with rasterio.open(path_mask,'r') as dst:
#             mask = dst.read()
#             mask = (mask[0]/255.0 > 0.5).astype(np.uint8)
#             print(np.unique(mask))

            

#             seq = create_seq_augment()
#             img,mask2 = agument(image1,mask,seq)

#     # # imgplot = plt.imshow(img_bin)
#     # # plt.show()
#     # # code for displaying multiple images in one figure

#     # #import libraries

#             print(np.unique(mask))
#             # create figure
#             fig = plt.figure(figsize=(10, 7))

#             # setting values to rows and column variables
#             rows = 2
#             columns = 2



#             # Adds a subplot at the 1st position
#             fig.add_subplot(rows, columns, 1)

#             # showing image
#             plt.imshow(image1)


#             # Adds a subplot at the 2nd position
#             fig.add_subplot(rows, columns, 2)

#             # showing image
#             plt.imshow(mask)


#             # Adds a subplot at the 3rd position
#             fig.add_subplot(rows, columns, 3)

#             # showing image
#             plt.imshow(img)

#             # Adds a subplot at the 4th position
#             fig.add_subplot(rows, columns, 4)

#             # showing image
#             plt.imshow(mask2)
#             plt.show()



def agument(image,augmentation=None):
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
            
            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            image2 = det.augment_image(image)
            # Change mask to np.uint8 because imgaug doesn't support np.bool
            
            # Verify that shapes didn't change
            assert image2.shape == image_shape, "Augmentation shouldn't change image size"
            
        except Exception:
            
            image2 = image
    else:
        
        image2 = image
    return image2

path_img = '/home/skymap/data/Bahrain_change/allqc/label_bong_nha_lech/test_code_shp/img/4.tif'

print('111111111111111111111111111')
with rasterio.open(path_img,'r') as src:
    image = src.read()

    image1 = image.transpose(1,2,0)
    
    print('22222222222222222222222222222222')
    seq = create_seq_augment()
    img = agument(image1,seq)
    print('44444444444444444444')
# # imgplot = plt.imshow(img_bin)
# # plt.show()
# # code for displaying multiple images in one figure

# #import libraries

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    # create figure
    # fig = plt.figure(figsize=(10, 7))

    # # setting values to rows and column variables
    # rows = 2
    # columns = 2

    print('5555555555555555555555')

    # Adds a subplot at the 1st position
    # fig.add_subplot(rows, columns, 1)

    # showing image
    a = plt.imshow(img)
    # b =plt.imshow(img)
    plt.show()
    # Adds a subplot at the 2nd position



    # Adds a subplot at the 3rd position
    # fig.add_subplot(rows, columns, 2)

    # showing image
    # b =plt.imshow(img)

    # Adds a subplot at the 4th position
    