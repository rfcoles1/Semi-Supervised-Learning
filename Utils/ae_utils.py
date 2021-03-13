import tensorflow as tf


def weighted_recon_loss(weight):
    def loss(y_true, y_pred):
        return tf.math.reduce_mean(((y_pred - y_true)*weight)**2)
    return loss

