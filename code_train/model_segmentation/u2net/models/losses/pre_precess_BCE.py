import tensorflow as tf

bce = tf.keras.losses.BinaryCrossentropy()
def pre_process_binary_cross_entropy(label, inputs):
    y = label
    loss = 0
    w_loss=1.0
    tmp_y = tf.cast(y, dtype=tf.float32)
    mask = tf.dtypes.cast(tmp_y > 0., tf.float32)
    b,h,w,c=mask.get_shape()
    positives = tf.math.reduce_sum(mask, axis=[1, 2, 3], keepdims=True)
    negatives = h*w*c-positives

    beta2 = positives / (negatives + positives)
    beta = negatives / (positives + negatives)
    pos_w = tf.where(tf.equal(y, 0.0), beta2, beta)
    # pos_w = tf.where(tf.equal(y, 0.0), beta, beta2)
    for tmp_p in inputs:
        l_cost = bce(y_true=tmp_y, y_pred=tmp_p, sample_weight=pos_w)
        loss += (l_cost*w_loss)
    return loss