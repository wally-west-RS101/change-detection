from tensorflow.keras import layers, Model, Input
import tensorflow as tf

def _upsample_like(src,tar):
    # src = tf.image.resize(images=src, size=tar.shape[1:3], method= 'bilinear')
    h = int(tar.shape[1]/src.shape[1])
    w = int(tar.shape[2]/src.shape[2])
    src = layers.UpSampling2D((h,w),interpolation='bilinear')(src)
    return src

def REBNCONV(x,out_ch=3,dirate=1):
    # x = layers.ZeroPadding2D(1*dirate)(x)
    x = layers.Conv2D(out_ch, 3, padding = "same", dilation_rate=1*dirate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x

def RSU7(hx, mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx, out_ch,dirate=1)

    hx1 = REBNCONV(hxin, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    hx2 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    hx3 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    hx4 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx4)

    hx5 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx5)

    hx6 = REBNCONV(hx, mid_ch,dirate=1)

    hx7 = REBNCONV(hx6, mid_ch,dirate=2)

    hx6d = REBNCONV(layers.concatenate([hx7,hx6]), mid_ch,dirate=1)
    hx6dup = _upsample_like(hx6d,hx5)

    hx5d = REBNCONV(layers.concatenate([hx6dup,hx5]), mid_ch,dirate=1)
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = REBNCONV(layers.concatenate([hx5dup,hx4]), mid_ch,dirate=1)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = REBNCONV(layers.concatenate([hx4dup,hx3]), mid_ch,dirate=1)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

    return hx1d + hxin

def RSU6(hx, mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx, out_ch,dirate=1)
    
    hx1 = REBNCONV(hxin, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    hx2 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    hx3 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    hx4 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx4)

    hx5 = REBNCONV(hx, mid_ch,dirate=1)

    hx6 = REBNCONV(hx, mid_ch,dirate=2)


    hx5d =  REBNCONV(layers.concatenate([hx6,hx5]), mid_ch,dirate=1)
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = REBNCONV(layers.concatenate([hx5dup,hx4]), mid_ch,dirate=1)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = REBNCONV(layers.concatenate([hx4dup,hx3]), mid_ch,dirate=1)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

    return hx1d + hxin

def RSU5(hx, mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx, out_ch,dirate=1)

    hx1 = REBNCONV(hxin, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    hx2 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    hx3 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    hx4 = REBNCONV(hx, mid_ch,dirate=1)

    hx5 = REBNCONV(hx4, mid_ch,dirate=2)

    hx4d = REBNCONV(layers.concatenate([hx5,hx4]), mid_ch,dirate=1)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = REBNCONV(layers.concatenate([hx4dup,hx3]), mid_ch,dirate=1)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

    return hx1d + hxin


def RSU4(hx,mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx, out_ch,dirate=1)

    hx1 = REBNCONV(hxin,mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    hx2 = REBNCONV(hx, mid_ch,dirate=1)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    hx3 = REBNCONV(hx, mid_ch,dirate=1)

    hx4 = REBNCONV(hx3, mid_ch,dirate=2)
    hx3d = REBNCONV(layers.concatenate([hx4,hx3]), mid_ch,dirate=1)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = REBNCONV(layers.concatenate([hx3dup,hx2]), mid_ch,dirate=1)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = REBNCONV(layers.concatenate([hx2dup,hx1]), out_ch,dirate=1)

    return hx1d + hxin

def RSU4F(hx, mid_ch=12, out_ch=3):
    hxin = REBNCONV(hx,out_ch,dirate=1)

    hx1 = REBNCONV(hxin, mid_ch,dirate=1)
    hx2 = REBNCONV(hx1, mid_ch,dirate=2)
    hx3 = REBNCONV(hx2, mid_ch,dirate=4)

    hx4 = REBNCONV(hx3, mid_ch,dirate=8)

    hx3d = REBNCONV(layers.concatenate([hx4,hx3]), mid_ch,dirate=4)
    hx2d = REBNCONV(layers.concatenate([hx3d,hx2]), mid_ch,dirate=2)
    hx1d = REBNCONV(layers.concatenate([hx2d,hx1]), out_ch,dirate=1)

    return hx1d + hxin

def U2NET(hx, out_ch=1):
    # hx = Input(shape=(480,480,3))
    #stage 1
    hx1 = RSU7(hx, 32,64)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    #stage 2
    hx2 = RSU6(hx, 32,128)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    #stage 3
    hx3 = RSU5(hx, 64,256)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    #stage 4
    hx4 = RSU4(hx, 128,512)
    hx = layers.MaxPool2D(2,strides=2)(hx4)

    #stage 5
    hx5 = RSU4F(hx, 256,512)
    hx = layers.MaxPool2D(2,strides=2)(hx5)

    #stage 6
    hx6 = RSU4F(hx, 256,512)
    hx6up = _upsample_like(hx6,hx5)

    #-------------------- decoder --------------------
    hx5d = RSU4F(layers.concatenate([hx6up,hx5]), 256,512)
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = RSU4(layers.concatenate([hx5dup,hx4]), 128,256)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = RSU5(layers.concatenate([hx4dup,hx3]), 64,128)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = RSU6(layers.concatenate([hx3dup,hx2]), 32,64)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = RSU7(layers.concatenate([hx2dup,hx1]), 16,64)


    #side output
    d1 = layers.Conv2D(1, 3,padding="same")(hx1d)

    d2 = layers.Conv2D(1, 3,padding="same")(hx2d)
    d2 = _upsample_like(d2,d1)

    d3 = layers.Conv2D(1, 3,padding="same")(hx3d)
    d3 = _upsample_like(d3,d1)

    d4 = layers.Conv2D(1, 3,padding="same")(hx4d)
    d4 = _upsample_like(d4,d1)

    d5 = layers.Conv2D(1, 3,padding="same")(hx5d)
    d5 = _upsample_like(d5,d1)

    d6 = layers.Conv2D(1, 3,padding="same")(hx6)
    d6 = _upsample_like(d6,d1)

    d0 = layers.Conv2D(out_ch,1)(layers.concatenate([d1,d2,d3,d4,d5,d6]))

    o1    = layers.Activation('sigmoid')(d1)
    o2    = layers.Activation('sigmoid')(d2)
    o3    = layers.Activation('sigmoid')(d3)
    o4    = layers.Activation('sigmoid')(d4)
    o5    = layers.Activation('sigmoid')(d5)
    o6    = layers.Activation('sigmoid')(d6)
    ofuse = layers.Activation('sigmoid')(d0)

    return [ofuse, o1, o2, o3, o4, o5, o6]

def U2NETP(hx, out_ch=1):
    # hx = Input(shape=(480,480,3))
    #stage 1
    hx1 = RSU7(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx1)

    #stage 2
    hx2 = RSU6(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx2)

    #stage 3
    hx3 = RSU5(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx3)

    #stage 4
    hx4 = RSU4(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx4)

    #stage 5
    hx5 = RSU4F(hx, 16,64)
    hx = layers.MaxPool2D(2,strides=2)(hx5)

    #stage 6
    hx6 = RSU4F(hx, 16,64)
    hx6up = _upsample_like(hx6,hx5)

    #-------------------- decoder --------------------
    hx5d = RSU4F(layers.concatenate([hx6up,hx5]), 16,64)
    hx5dup = _upsample_like(hx5d,hx4)

    hx4d = RSU4(layers.concatenate([hx5dup,hx4]), 16,64)
    hx4dup = _upsample_like(hx4d,hx3)

    hx3d = RSU5(layers.concatenate([hx4dup,hx3]), 16,64)
    hx3dup = _upsample_like(hx3d,hx2)

    hx2d = RSU6(layers.concatenate([hx3dup,hx2]), 16,64)
    hx2dup = _upsample_like(hx2d,hx1)

    hx1d = RSU7(layers.concatenate([hx2dup,hx1]), 16,64)


    #side output
    d1 = layers.Conv2D(1, 3,padding="same")(hx1d)

    d2 = layers.Conv2D(1, 3,padding="same")(hx2d)
    d2 = _upsample_like(d2,d1)

    d3 = layers.Conv2D(1, 3,padding="same")(hx3d)
    d3 = _upsample_like(d3,d1)

    d4 = layers.Conv2D(1, 3,padding="same")(hx4d)
    d4 = _upsample_like(d4,d1)

    d5 = layers.Conv2D(1, 3,padding="same")(hx5d)
    d5 = _upsample_like(d5,d1)

    d6 = layers.Conv2D(1, 3,padding="same")(hx6)
    d6 = _upsample_like(d6,d1)

    d0 = layers.Conv2D(out_ch,1)(layers.concatenate([d1,d2,d3,d4,d5,d6]))

    o1    = layers.Activation('sigmoid')(d1)
    o2    = layers.Activation('sigmoid')(d2)
    o3    = layers.Activation('sigmoid')(d3)
    o4    = layers.Activation('sigmoid')(d4)
    o5    = layers.Activation('sigmoid')(d5)
    o6    = layers.Activation('sigmoid')(d6)
    ofuse = layers.Activation('sigmoid')(d0)

    return tf.stack([ofuse, o1, o2, o3, o4, o5, o6])

def Model_U2Net(image_size, num_band):
    hx = Input(shape=(image_size,image_size,num_band))
    out = U2NET(hx)
    model = Model(inputs = hx, outputs = out)
    return model

def Model_U2Netp(image_size, num_band):
    hx = Input(shape=(image_size,image_size,num_band))
    out = U2NETP(hx)
    model = Model(inputs = hx, outputs = out)
    return model

if __name__ == '__main__':
    pass