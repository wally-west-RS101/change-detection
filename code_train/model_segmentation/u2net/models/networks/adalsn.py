from tensorflow.keras import layers, Model, regularizers, Input, backend
import tensorflow as tf

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False,
        name=conv_name)(x)
    # x = layers.BatchNormalization(axis=bn_axis, scale=False, center=False,name=bn_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x

def InceptionV3(img_input):
    output = []
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = layers.ZeroPadding2D(35)(img_input)
    x = conv2d_bn(x, 32, 3, 3, strides=(1, 1), padding='valid')
    x = conv2d_bn(x, 32, 3, 3)
    x = conv2d_bn(x, 64, 3, 3)
    output.append(x)

    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3)
    output.append(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1, padding='valid')

    branch5x5 = conv2d_bn(x, 48, 1, 1, padding='valid')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding='valid')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1, padding='valid')
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed0')

    # mixed 1: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1, padding='valid')

    branch5x5 = conv2d_bn(x, 48, 1, 1, padding='valid')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding='valid')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, padding='valid')
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed1')

    # mixed 2: 35 x 35 x 288
    branch1x1 = conv2d_bn(x, 64, 1, 1, padding='valid')

    branch5x5 = conv2d_bn(x, 48, 1, 1, padding='valid')
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding='valid')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1, padding='valid')
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed2')
    output.append(x)
    # mixed 3: 17 x 17 x 768
    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding = 'valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1, padding= 'valid')
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding = 'valid')

    # branch_pool = layers.ZeroPadding2D(1)(x)
    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

    # mixed 4: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, padding = 'valid')

    branch7x7 = conv2d_bn(x, 128, 1, 1, padding = 'valid')
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1, padding = 'valid')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding = 'valid')
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed4')

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1, padding = 'valid')

        branch7x7 = conv2d_bn(x, 160, 1, 1, padding = 'valid')
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1, padding = 'valid')
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = layers.AveragePooling2D( (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding = 'valid')
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed' + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = conv2d_bn(x, 192, 1, 1, padding = 'valid')

    branch7x7 = conv2d_bn(x, 192, 1, 1, padding = 'valid')
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1, padding = 'valid')
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding = 'valid')
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name='mixed7')
    output.append(x)

    # mixed 8: 8 x 8 x 1280
    branch3x3 = conv2d_bn(x, 192, 1, 1, padding = 'valid')
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1, padding = 'valid')
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1, padding = 'valid')

        branch3x3 = conv2d_bn(x, 384, 1, 1, padding = 'valid')
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1, padding = 'valid')
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1, padding = 'valid')
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed' + str(9 + i))
    output.append(x)

    return output

def crop(d, region):
    x, y, h, w = region
    d1 = d[:, x:x + h, y:y + w, :]
    return d1

def DilConv(x, kernel_size, padding, dilation, stride = 1, C_out = 64):
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding)(x)
    x = layers.Conv2D(C_out, kernel_size=kernel_size, strides=stride, dilation_rate=dilation)(x)
    return x

def Conv(x, kernel_size, padding, stride = 1, C_out = 64):
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding)(x)
    x = layers.Conv2D(C_out, kernel_size=kernel_size, strides=stride)(x)
    return x

def Identity(x):
    return x

def cell1(x, flag=1):
    x1 = Conv(x, 5, 2)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def cell2(x, flag=1):
    x1 = DilConv(x, 3, 2, 2)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def cell3(x, flag=1):
    x1 = Conv(x, 3, 1)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def cell4(x, flag=1):
    x1 = DilConv(x, 5, 8, 4)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def cell_fuse(x, flag=1):
    x1 = DilConv(x, 3, 2, 2)
    out_x = x + x1
    if flag == 1:
        out_x = layers.UpSampling2D(interpolation = 'bilinear')(out_x)
    return out_x

def Adalsn(image_size, shape, C = 64):
    # input = Input(shape=(299,299,shape))
    input = Input(shape=(image_size,image_size,shape))
    size = input.shape[1:3]
    conv1, conv2, conv3, conv4, conv5 = InceptionV3(input)
    dsn1 = layers.Conv2D(C, 1)(conv1)
    dsn2 = layers.Conv2D(C, 1)(conv2)
    dsn3 = layers.Conv2D(C, 1)(conv3)
    dsn4 = layers.Conv2D(C, 1)(conv4)
    dsn5 = layers.Conv2D(C, 1)(conv5)
    c1 = cell1(dsn5)

    mm1 = layers.concatenate([c1, crop(dsn4, (0, 0)+ c1.shape[1:3])])
    mm1 = layers.Conv2D(C, 1)(mm1)
    d4_2 = layers.Activation('relu')(mm1)
    c2 = cell2(d4_2)

    mm2 = layers.UpSampling2D(interpolation= 'bilinear')(c1)
    mm2 = layers.concatenate([mm2, crop(dsn3, (0, 0) + mm2.shape[1:3])])
    mm2 = layers.Conv2D(C, 1)(mm2)
    d3_2 = layers.Activation('relu')(mm2)
    d3_2 = layers.concatenate([c2, crop(d3_2, (0, 0) + c2.shape[1:3])])
    d3_2 = layers.Conv2D(C, 1)(d3_2)
    d3_2 = layers.Activation('relu')(d3_2)
    c3 = cell3(d3_2)

    c4 = cell4(dsn2)

    d_fuse = tf.zeros_like(c3)
    d_fuse = crop(layers.UpSampling2D(interpolation="bilinear")(c2), (0,0) + d_fuse.shape[1:3]) + crop(c3, (0, 0) + c3.shape[1:3]) + crop(layers.MaxPool2D()(c4), (0, 0) + d_fuse.shape[1:3])
    d_fuse = cell_fuse(d_fuse)
    d_fuse = layers.Conv2D(1, 1)(d_fuse)
    d_fuse = layers.ZeroPadding2D(7)(d_fuse)
    sss = layers.Conv2D(1, 15)(d_fuse)

    out_fuse = crop(sss, (34, 34) + size)
    out = crop(layers.Conv2D(1, 1)(layers.UpSampling2D(size=(4, 4), interpolation = 'bilinear')(c2)), (34, 34) + size)

    out_fuse = layers.Activation('sigmoid')(out_fuse)
    out = layers.Activation('sigmoid')(out)
    model = Model(input, [out_fuse, out], name='inception_v3')
    # model.summary()
    return model
 