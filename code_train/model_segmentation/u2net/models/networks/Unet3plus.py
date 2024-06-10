from tensorflow.keras import layers, Model, Input
import tensorflow as tf
from .u2net import RSU4F, RSU4, RSU5, RSU6, RSU7


def UnetConv2(inputs, filters, kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), n = 2, name = ''):
    Z = inputs
    for i in range(1, n+1):
        Z = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides=strides, padding = "same",
                        kernel_initializer = kernel_initializer, name = f"{name}_unetconv2_conv{i}")(Z)
        Z = layers.BatchNormalization(axis = -1, name = f"{name}_unetconv2_bn{i}")(Z)
        Z = layers.Activation('relu', name=f"{name}_unetconv2_relu{i}")(Z)
    return Z

def Unet3plus_deep_supervision(inputs, upsample_size, filters = 1, kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), name = ''):
    Z = inputs
    Z = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides=strides, padding = "same",
                    kernel_initializer = kernel_initializer, name = f"{name}_ds_conv")(Z)

    Z = layers.UpSampling2D(size = upsample_size, interpolation = 'bilinear', name = f"{name}_ds_bilinear_upsample")(Z)
    Z = layers.Activation('sigmoid', dtype= 'float32', name=f"{name}_sigmoid")(Z)
    return Z

def FullScaleBlock(inputs, num_smaller_scale, filters = 64, kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), first_conv2d_strides = (1, 1), name = ''):
    inp_d = inputs
    for i in range(num_smaller_scale):
        inp_d[i] = layers.MaxPool2D(pool_size = 2 ** (num_smaller_scale - i), name = f"{name}_fsb_maxpool_{i}")(inp_d[i])
        inp_d[i] = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same',  
                                 kernel_initializer = kernel_initializer, name = f"{name}_fsb_conv_{i}")(inp_d[i])
        inp_d[i] = layers.BatchNormalization(axis = -1, name = f"{name}_fsb_bn_{i}")(inp_d[i])
        inp_d[i] = layers.ReLU(name = f"{name}_fsb_relu_{i}")(inp_d[i])

    inp_d[num_smaller_scale] = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same',
                                             kernel_initializer = kernel_initializer, name = f"{name}_fsb_conv_{num_smaller_scale}")(inp_d[num_smaller_scale])
    inp_d[num_smaller_scale] = layers.BatchNormalization(axis = -1, name = f"{name}_fsb_bn_{num_smaller_scale}")(inp_d[num_smaller_scale])
    inp_d[num_smaller_scale] = layers.ReLU(name = f"{name}_fsb_relu_{num_smaller_scale}")(inp_d[num_smaller_scale])

    for i in range(num_smaller_scale + 1, 5):
        inp_d[i] = layers.UpSampling2D(size = 2 ** (i - num_smaller_scale), interpolation = 'bilinear', name = f"{name}_fsb_bilinear_upsample_{i}")(inp_d[i])
        inp_d[i] = layers.Conv2D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same',
                                 kernel_initializer = kernel_initializer, name = f"{name}_fsb_conv_{i}")(inp_d[i])
        inp_d[i] = layers.BatchNormalization(axis = -1, name = f"{name}_fsb_bn_{i}")(inp_d[i])
        inp_d[i] = layers.ReLU(name = f"{name}_fsb_relu_{i}")(inp_d[i])
        
    Z = layers.Concatenate(axis = -1, name = f"{name}_fsb_concat")(inp_d)
    Z = layers.Conv2D(filters = 320, kernel_size = kernel_size, strides = strides, padding = 'same',
                     kernel_initializer = kernel_initializer, name = f"{name}_fsb_fusion_conv")(Z)
    Z = layers.BatchNormalization(axis = -1, name = f"{name}_fusion_bn")(Z)
    Z = layers.ReLU(name = f"{name}_fusion_relu")(Z)

    Zs = layers.Conv2D(filters = 1, kernel_size = (3, 3), strides = strides, padding = 'same',
                        kernel_initializer = kernel_initializer, name = f"{name}_fsb_sd_conv")(Z)
    
    if num_smaller_scale != 0:
        Zs = layers.UpSampling2D(size = 2 ** num_smaller_scale, interpolation = 'bilinear', name = f"{name}_fsb_sd_bilinear_upsample")(Zs)
    Zs = layers.Activation('sigmoid', dtype = 'float32', name = f"{name}_sd_sigmoid")(Zs)
    
    return Z, Zs


def Unet3plus(inputs, kernel_initializer = "he_normal", encoder_conv_n = 2):
    Z = inputs
    ## conv1    
#     Z1 = UnetConv2(Z, 64, name = 'conv1', kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), n = encoder_conv_n)
    Z1 = RSU7(Z, 16,32)
    Zo = layers.MaxPool2D(pool_size = 2, name = f"conv1_maxpool")(Z1)
    
    ## conv2    
#     Z2 = UnetConv2(Zo, 128, name = 'conv2', kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), n = encoder_conv_n)
    Z2 = RSU6(Zo, 32,64)
    Zo = layers.MaxPool2D(pool_size = 2, name = f"conv2_maxpool")(Z2)
    
    ## conv3    
#     Z3 = UnetConv2(Zo, 256, name = 'conv3', kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), n = encoder_conv_n)
    Z3 = RSU5(Zo, 64,128)
    Zo = layers.MaxPool2D(pool_size = 2, name = f"conv3_maxpool")(Z3)
    
    ## conv4    
#     Z4 = UnetConv2(Zo, 512, name = 'conv4', kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), n = encoder_conv_n)
    Z4 = RSU4(Zo, 128,256)
    Zo = layers.MaxPool2D(pool_size = 2, name = f"conv4_maxpool")(Z4)
    
    ## conv5    
#     Zd5 = UnetConv2(Zo, 1024, name = 'conv5', kernel_size = (3, 3), kernel_initializer = "he_normal", strides = (1, 1), n = encoder_conv_n)
    Zd5 = RSU4F(Zo, 256,512)
    
    Zsd5 = Unet3plus_deep_supervision(Zd5, upsample_size = 2 ** 4, name = "enc5_ds")
    
    # CGM
#     Z = layers.SpatialDropout2D(rate = 0.5)(Zd5)
#     Z = layers.Conv2D(filters = 1, kernel_size = (1,1), strides = (1, 1), padding = 'same', kernel_initializer = kernel_initializer, name = f"cgm_conv")(Z)
#     Z = tfa.layers.AdaptiveMaxPooling2D(output_size = 1, name = "cgm_adaptive_pool")(Z)
#     Zco = layers.Activation('sigmoid', name = 'cgm_sigmoid', dtype='float32')(Z)
#     Zc = tf.where(Zco > 0.5, 1., 0.)
#     Zco = tf.squeeze(Zco, axis = [-3, -2, -1], name = "Zco_squeeze")
#     Zsd5 = tf.multiply(Zsd5, Zc, name = "enc5_multiply")
    
    ## dec4
    Zd4, Zsd4 = FullScaleBlock([Z1, Z2, Z3, Z4, Zd5], num_smaller_scale = 3, name = "dec4")
#     Zsd4 = tf.multiply(Zsd4, Zc, name = "enc4_multiply")

    ## dec3
    Zd3, Zsd3 = FullScaleBlock([Z1, Z2, Z3, Zd4, Zd5], num_smaller_scale = 2, name = "dec3") 
#     Zsd3 = tf.multiply(Zsd3, Zc, name = "enc3_multiply")
    
    ## dec2
    Zd2, Zsd2 = FullScaleBlock([Z1, Z2, Zd3, Zd4, Zd5], num_smaller_scale = 1, name = "dec2") 
#     Zsd2 = tf.multiply(Zsd2, Zc, name = "enc2_multiply")
    
    ## dec1
    Zd1, Zsd1 = FullScaleBlock([Z1, Zd2, Zd3, Zd4, Zd5], num_smaller_scale = 0, name = "dec1") 
#     Zsd1 = tf.multiply(Zsd1, Zc, name = "enc1_multiply")
    
    return Zsd1, Zsd2, Zsd3, Zsd4, Zsd5

def Model_UNet3plus(image_size, num_band):
    hx = Input(shape=(image_size,image_size,num_band))
    out = Unet3plus(hx)
    model = Model(inputs = hx, outputs = out)
    # model.summary()
    return model

if __name__ == '__main__':
    model = Model_UNet3plus(512, 3)
    model.summary()