import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers, Input


l2 = regularizers.l2
w_decay=1e-3
weight_init = tf.initializers.glorot_uniform()

def DoubleConvBlock(input, mid_features, out_features=None, stride=(1,1), use_bn=True,use_act=True):
    out_features = mid_features if out_features is None else out_features
    k_reg = None if w_decay is None else l2(w_decay)
    x = layers.Conv2D(filters=mid_features, kernel_size=(3, 3), strides=stride, padding='same', kernel_initializer=weight_init, kernel_regularizer=k_reg)(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=out_features, kernel_size=(3, 3), strides=(1,1), padding='same', kernel_initializer=weight_init, kernel_regularizer=k_reg)(x)
    x = layers.BatchNormalization()(x)
    if use_act:
        x = layers.ReLU()(x)
    return x

def SingleConvBlock(input, out_features, k_size=(1,1),stride=(1,1), use_bs=False, use_act=False, w_init=None):
    k_reg = None if w_decay is None else l2(w_decay)
    x = layers.Conv2D(filters=out_features, kernel_size=k_size, strides=stride, padding='same',kernel_initializer=w_init, kernel_regularizer=k_reg)(input)
    if use_bs:
        x = layers.BatchNormalization()(x)
    if use_act:
        x = layers.ReLU()(x)
    return x

def UpConvBlock(input_data, up_scale):
    total_up_scale = 2 ** up_scale
    constant_features = 16
    k_reg = None if w_decay is None else l2(w_decay)
    features = []
    for i in range(up_scale):
        out_features = 1 if i == up_scale-1 else constant_features
        if i==up_scale-1:
            input_data = layers.Conv2D(filters=out_features, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu', kernel_initializer=tf.initializers.TruncatedNormal(mean=0.), kernel_regularizer=k_reg,use_bias=True)(input_data)
            input_data = layers.Conv2DTranspose(out_features, kernel_size=(total_up_scale,total_up_scale), strides=(2,2), padding='same', kernel_initializer=tf.initializers.TruncatedNormal(stddev=0.1), kernel_regularizer=k_reg,use_bias=True)(input_data)
        else:
            input_data = layers.Conv2D(filters=out_features, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu',kernel_initializer=weight_init, kernel_regularizer=k_reg,use_bias=True)(input_data)
            input_data = layers.Conv2DTranspose(out_features, kernel_size=(total_up_scale,total_up_scale),strides=(2,2), padding='same', use_bias=True, kernel_initializer=weight_init, kernel_regularizer=k_reg)(input_data)
    return input_data

def _DenseLayer(inputs, out_features):
    k_reg = None if w_decay is None else l2(w_decay)
    x, x2 = tuple(inputs)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=out_features, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=weight_init, kernel_regularizer=k_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=out_features, kernel_size=(3,3), strides=(1,1), padding='same', kernel_initializer=weight_init, kernel_regularizer=k_reg)(x)
    x = layers.BatchNormalization()(x)
    return 0.5 * (x + x2), x2

def _DenseBlock(input_da, num_layers, out_features):
    for i in range(num_layers):
        input_da = _DenseLayer(input_da, out_features)
    return input_da

def DexiNed(image_size, image_band_channel):
    img_input = Input(shape=(image_size,image_size,image_band_channel), name='input')

    block_1 = DoubleConvBlock(img_input, 32, 64, stride=(2,2),use_act=False)
    block_1_side = SingleConvBlock(block_1, 128, k_size=(1,1),stride=(2,2),use_bs=True, w_init=weight_init)

    # Block 2
    block_2 = DoubleConvBlock(block_1, 128, use_act=False)
    block_2_down = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(block_2)
    block_2_add = block_2_down + block_1_side
    block_2_side = SingleConvBlock(block_2_add, 256,k_size=(1,1),stride=(2,2),use_bs=True, w_init=weight_init)

    # Block 3
    block_3_pre_dense = SingleConvBlock(block_2_down,256,k_size=(1,1),stride=(1,1),use_bs=True,w_init=weight_init)
    block_3, _ = _DenseBlock([block_2_add, block_3_pre_dense], 2, 256)
    block_3_down = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(block_3)
    block_3_add = block_3_down + block_2_side
    block_3_side = SingleConvBlock(block_3_add, 512,k_size=(1,1),stride=(2,2),use_bs=True,w_init=weight_init)

    # Block 4
    block_4_pre_dense_256 = SingleConvBlock(block_2_down, 256,k_size=(1,1),stride=(2,2), w_init=weight_init)
    block_4_pre_dense = SingleConvBlock(block_4_pre_dense_256 + block_3_down, 512,k_size=(1,1),stride=(1,1),use_bs=True, w_init=weight_init)
    block_4, _ = _DenseBlock([block_3_add, block_4_pre_dense], 3, 512)
    block_4_down = layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(block_4)
    block_4_add = block_4_down + block_3_side
    block_4_side = SingleConvBlock(block_4_add, 512,k_size=(1,1),stride=(1,1),use_bs=True, w_init=weight_init)

    # Block 5
    block_5_pre_dense_512 = SingleConvBlock(block_4_pre_dense_256, 512, k_size=(1,1),stride=(2,2), w_init=weight_init)
    block_5_pre_dense = SingleConvBlock(block_5_pre_dense_512 + block_4_down, 512,k_size=(1,1),stride=(1,1),use_bs=True, w_init=weight_init)
    block_5, _ = _DenseBlock([block_4_add, block_5_pre_dense], 3, 512)
    block_5_add = block_5 + block_4_side

    # Block 6
    block_6_pre_dense = SingleConvBlock(block_5, 256,k_size=(1,1),stride=(1,1),use_bs=True, w_init=weight_init)
    block_6, _ =  _DenseBlock([block_5_add, block_6_pre_dense], 3, 256)


    out_1 = UpConvBlock(block_1, 1)
    out_2 = UpConvBlock(block_2, 1)
    out_3 = UpConvBlock(block_3, 2)
    out_4 = UpConvBlock(block_4, 3)
    out_5 = UpConvBlock(block_5, 4)
    out_6 = UpConvBlock(block_6, 4)

    # concatenate multiscale outputs
    block_cat = tf.concat([out_1, out_2, out_3, out_4, out_5, out_6], 3)  # BxHxWX6
    block_cat = SingleConvBlock(block_cat, 1,k_size=(1,1),stride=(1,1), w_init=tf.constant_initializer(1/5))  # BxHxWX1
    
    block_cat = layers.Activation('sigmoid')(block_cat)
    out_1 = layers.Activation('sigmoid')(out_1)
    out_2 = layers.Activation('sigmoid')(out_2)
    out_3 = layers.Activation('sigmoid')(out_3)
    out_4 = layers.Activation('sigmoid')(out_4)
    out_5 = layers.Activation('sigmoid')(out_5)
    out_6 = layers.Activation('sigmoid')(out_6)

    model = Model(inputs=[img_input], outputs=[block_cat, out_1, out_2, out_3, out_4, out_5, out_6])
    # model.summary()
    return model
