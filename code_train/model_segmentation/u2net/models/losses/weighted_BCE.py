import tensorflow as tf

def _to_tensor(x, dtype):
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x

def weighted_cross_entropy_loss(label, inputs):
    loss = 0
    y = tf.cast(label, dtype=tf.float32)
    negatives = tf.math.reduce_sum(1.-y)
    positives = tf.math.reduce_sum(y)
    beta = negatives/(negatives + positives)
    pos_w = beta/(1-beta)

    for predict in inputs:
        _epsilon = _to_tensor(tf.keras.backend.epsilon(), predict.dtype.base_dtype)
        predict   = tf.clip_by_value(predict, _epsilon, 1 - _epsilon)
        predict   = tf.math.log(predict/(1 - predict))
        
        cost = tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=predict, pos_weight=pos_w)
        cost = tf.math.reduce_mean(cost*(1-beta))
        loss += tf.where(tf.equal(positives, 0.0), 0.0, cost)
    return loss

def weighted_cross_entropy_loss_test(label, inputs):
    y = tf.cast(label, dtype=tf.float32)
    negatives = tf.math.reduce_sum(1.-y)
    positives = tf.math.reduce_sum(y)
    beta = negatives/(negatives + positives)
    pos_w = beta/(1-beta)
    loss = tf.Variable(tf.zeros(len(inputs), tf.float32))
    
    for idx, predict in enumerate(inputs):
        _epsilon = _to_tensor(tf.keras.backend.epsilon(), predict.dtype.base_dtype)
        predict   = tf.clip_by_value(predict, _epsilon, 1 - _epsilon)
        predict   = tf.math.log(predict/(1 - predict))
        
        cost = tf.nn.weighted_cross_entropy_with_logits(labels=y, logits=predict, pos_weight=pos_w)
        cost = tf.math.reduce_mean(cost*(1-beta))
        loss = loss[idx].assign(tf.where(tf.equal(positives, 0.0), 0.0, cost))
    return tf.math.reduce_sum(loss)