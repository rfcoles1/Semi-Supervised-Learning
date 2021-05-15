import warnings 
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def residual(y_true, y_pred):
    return (y_true - y_pred)/(1+y_true)

def bias(y_true, y_pred):
    resid = residual(y_true, y_pred)
    return tf.reduce_mean(abs(resid))
    
def stdev(y_true, y_pred):
    resid = residual(y_true, y_pred)
    return tf.math.reduce_std(abs(resid))

def MAD(y_true, y_pred):  
    resid = residual(y_true, y_pred)
    median = tfp.stats.percentile(resid, 50.0, interpolation='midpoint')
    return tfp.stats.percentile(abs(resid - median), 50.0, interpolation='midpoint')

#TODO all currently get average by batch, should that be done for outliers?
def outliers(y_true, y_pred):
    resid = residual(y_true, y_pred)
    median = tfp.stats.percentile(resid, 50.0, interpolation='midpoint')
    mad = tfp.stats.percentile(abs(resid - median), 50.0, interpolation='midpoint')
    bool_outliers = tf.math.greater(abs(resid), 5*(1.4826*mad))
    return tf.reduce_sum(tf.cast(bool_outliers, tf.float32))

def bias_MAD(y_true, y_pred):
    resid = residual(y_true, y_pred)
    median = tfp.stats.percentile(resid, 50.0, interpolation='midpoint')
    return tf.sqrt(tf.reduce_mean(abs(resid))) + 1.4826 * (tf.reduce_mean(abs(resid - median)))
