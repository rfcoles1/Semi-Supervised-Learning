import warnings 
warnings.filterwarnings('ignore')

import tensorflow as tf
import tensorflow_probability as tfp

def abs_bias_loss(y_true, y_pred):
    return tf.reduce_mean(abs(y_true - y_pred))/(1 + y_true)
    
def MAD_loss(y_true, y_pred):  
    resid = (y_true - y_pred)/(1 + y_true)
    #median = tf.contrib.distributions.percentile(resid,50.)
    median = tfp.stats.percentile(resid, 50.0, interpolation='midpoint')
    return tf.reduce_mean(abs(resid - median))

def bias_MAD_loss(y_true, y_pred):
    resid = (y_true - y_pred)/(1 + y_true)
    #median = tf.contrib.distributions.percentile(resid,50.)
    median = tfp.stats.percentile(resid, 50.0, interpolation='midpoint')
    return tf.sqrt(tf.reduce_mean(abs(resid))) + 1.4826 * (tf.reduce_mean(abs(resid - median)))

