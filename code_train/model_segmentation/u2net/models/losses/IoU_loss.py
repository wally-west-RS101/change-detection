import tensorflow as tf

def IoULoss(calc_axis = (-3, -2, -1), smooth = 1e-8):
    def IoULoss_(y_true, y_pred):
        y_true = tf.cast(y_true, y_pred.dtype)
        intersection = tf.reduce_sum(tf.multiply(y_true, y_pred), axis = calc_axis)
        total = tf.reduce_sum(y_true, axis = calc_axis) + tf.reduce_sum(y_pred, axis = calc_axis)
        union = total - intersection       
        IoU = (intersection + smooth) / (union + smooth)   
        return 1 - IoU
    return IoULoss_

def total_IoULoss(y_true, y_pred):
    loss = 0
    for pre in y_pred:
        loss += IoULoss()(y_true, pre)
    return loss