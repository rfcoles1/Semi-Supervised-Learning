import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers


def elu_plus(x):
    return K.elu(x) + 1 + 1e-6

def get_mix(y_pred, num_mixes):
    out_mu, out_sigma, out_pi = tf.split(y_pred,\
        num_or_size_splits=[num_mixes, num_mixes, num_mixes], axis=-1)
    mus = tf.split(out_mu, num_or_size_splits=num_mixes, axis=1)
    sigs = tf.split(out_sigma, num_or_size_splits=num_mixes, axis=1)

    cat = tfd.Categorical(logits=out_pi)
    coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) \
        for loc, scale in zip(mus, sigs)]

    mixture = tfd.Mixture(cat=cat, components=coll)
    return mixture

def _mdn_loss(y_true, y_pred):
    mixture = get_mix(y_pred, num_mixes)

    loss = mixture.log_prob(y_true)
    loss = tf.negative(loss)
    return tf.reduce_mean(loss)

def mdn_loss(num_mixes):
    def loss(y_true, y_pred):
        mixture = get_mix(y_pred, num_mixes)
        
        loss = mixture.log_prob(y_true)
        loss = tf.negative(loss)
        return tf.reduce_mean(loss)
    return loss

