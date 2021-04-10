import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import scipy
import tensorflow as tf
import tensorflow_probability as tfp

def residual(y_true, y_pred):
    return (y_true - y_pred)/(1+y_true)

def MAD(y_true, y_pred):
    resid = residual(y_true, y_pred)
    return 1.4826*np.median(np.abs(resid - np.median(resid)))

def outliers(y_true, y_pred):
    resid = residual(y_true, y_pred)
    mad = MAD(y_true, y_pred)
    return sum(np.abs(resid) > (5*mad))


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

