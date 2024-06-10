import tensorflow as tf

def ms_ssim(y_true, y_pred, **kwargs):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    y_pred = tf.squeeze(y_pred)
    y_true = tf.squeeze(y_true)
    
    tf_ms_ssim = tf.image.ssim_multiscale(y_true, y_pred, **kwargs)
        
    return 1 - tf_ms_ssim

class MS_SSIM_Loss(tf.keras.losses.Loss):
    def __init__(self, calc_axis = (-3, -2, -1), max_val = 1.0, **kwargs):
        super().__init__(**kwargs)
        
        self.calc_axis = calc_axis
        self.max_val = max_val
        
        
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        ms_ssim = tf.image.ssim_multiscale(y_true, y_pred, max_val = self.max_val)
        
        return 1 - ms_ssim
