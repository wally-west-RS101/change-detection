import tensorflow as tf
import tensorflow.keras.backend as K

def tversky_coef(y_true, y_pred, alpha=0.5, const=K.epsilon()):
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])
    
    true_pos  = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.reduce_sum((1-y_true_pos) * y_pred_pos)
    
    coef_val = (true_pos + const)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + const)
    
    return coef_val

def tversky(y_true, y_pred, alpha=0.5, const=K.epsilon()):

    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    loss_val = 1 - tversky_coef(y_true, y_pred, alpha=alpha, const=const)
    
    return loss_val

def focal_tversky(y_true, y_pred, alpha=0.5, gamma=4/3, const=K.epsilon()):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    loss_val = tf.math.pow((1-tversky_coef(y_true, y_pred, alpha=alpha, const=const)), 1/gamma)
    
    return loss_val