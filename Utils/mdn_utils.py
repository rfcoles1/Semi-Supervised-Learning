import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions

import metrics as metric

def elu_plus(x):
    return K.elu(x) + 1 + 1e-6

def get_mix(y_pred, num_mixes):
    out_mu, out_sigma, out_pi = tf.split(y_pred,\
        num_or_size_splits=[num_mixes, num_mixes, num_mixes], axis=-1)

    mixture = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits = out_pi),
        components_distribution=tfd.Normal(
            loc = out_mu, scale= out_sigma))
    return mixture

#TODO improve this function to be quicker, 
def get_mode(y_pred, num_mixes, n_classes):
    rnge = tf.cast(tf.reshape(tf.linspace(0,1,n_classes),(n_classes,-1)),tf.float32)
    mix = get_mix(y_pred, num_mixes)
    probs = mix.prob(rnge)
    probs = probs/tf.reduce_sum(probs)
    idx = tf.math.argmax(probs)
    mode = tf.gather(rnge, idx)
    return mode 


def mdn_loss(num_mixes):
    def loss(y_true, y_pred):
        mixture = get_mix(y_pred, num_mixes)
        
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        return tf.reduce_mean(loss)
    return loss

def mdn_bias(num_mixes, num_classes):
    def bias(y_true, y_pred_mix):
        y_pred = get_mode(y_pred_mix, num_mixes, num_classes)
        return metric.bias(y_true, y_pred)
    return bias

def mdn_stdev(num_mixes, num_classes):
    def stdev(y_true, y_pred_mix):
        y_pred = get_mode(y_pred_mix, num_mixes, num_classes)
        return metric.stdev(y_true, y_pred)
    return stdev

def mdn_MAD(num_mixes, num_classes):
    def MAD(y_true, y_pred_mix):
        y_pred = get_mode(y_pred_mix, num_mixes, num_classes)
        return metric.MAD(y_true, y_pred)
    return MAD

def mdn_outliers(num_mixes, num_classes):
    def outliers(y_true, y_pred_mix):
        y_pred = get_mode(y_pred_mix, num_mixes, num_classes)
        return metric.outliers(y_true, y_pred)
    return outliers

def mdn_bias_MAD(num_mixes, num_classes):
    def bias_MAD(y_true, y_pred_mix):
        y_pred = get_mode(y_pred_mix, num_mixes, num_classes)
        return metric.bias_MAD(y_true, y_pred)
    return bias_MAD
