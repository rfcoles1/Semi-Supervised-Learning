from Networks import *

import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers

sys.path.insert(1, '../Utils')
from metrics import *
from datasets import *
from augment import *

NUM_ = 5

def mdn_loss(y_true, y_pred):
    out_mu, out_sigma, out_pi = tf.split(y_pred,\
        num_or_size_splits=[NUM_, NUM_, NUM_], axis=-1)
    mus = tf.split(out_mu, num_or_size_splits=NUM_, axis=1)
    sigs = tf.split(out_sigma, num_or_size_splits=NUM_, axis=1)
    cat = tfd.Categorical(logits=out_pi)
    coll = [tfd.MultivariateNormalDiag(loc=loc,scale_diag=scale) \
        for loc, scale in zip(mus, sigs)]
    mixture = tfd.Mixture(cat=cat, components=coll)

    loss = mixture.log_prob(y_true)
    loss = tf.negative(loss)
    return tf.reduce_mean(loss)

def elu_plus(x):
    return K.elu(x) + 1 + 1e-6

def get_mix(y_pred):
    out_mu, out_sigma, out_pi = tf.split(y_pred,\
        num_or_size_splits=[NUM_, NUM_, NUM_], axis=-1)
    mus = tf.split(out_mu, num_or_size_splits=NUM_, axis=1)
    sigs = tf.split(out_sigma, num_or_size_splits=NUM_, axis=1)
    cat = tfd.Categorical(logits=out_pi)
    coll = [tfd.MultivariateNormalDiag(loc=loc,scale_diag=scale) \
        for loc, scale in zip(mus, sigs)]
    mixture = tfd.Mixture(cat=cat, components=coll)
    
    return mixture

class MDN(Network):
    def __init__(self, input_shape, noise=False):
        super().__init__()
  
        self.dirpath = 'records_z/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        
        self.batch_size = 64
        self.input_shape = input_shape
        self.num_out = 1
        self.num_mixtures = NUM_

        lr = 1e-6
        optimizer = keras.optimizers.Adam(lr=lr)            

        self.inp = layers.Input(self.input_shape)
        self.Net = tf.keras.models.Model(self.inp, self.network_mdn(self.inp))
        self.Net.compile(loss = mdn_loss,  optimizer=optimizer)


    def network_mdn(self,x):
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,\
            input_shape=self.input_shape)
        base_model.trainable = True
        x = base_model(x, training=True)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(512, activation = 'relu')(x)
        x = layers.Dense(256, activation = 'relu')(x)
        x = layers.Dense(128, activation = 'relu')(x)
        
        mus = layers.Dense(self.num_mixtures, name='mus')(x)

        #std must be greater than 0, #try exp, softplus, or elu + 1
        sigmas = layers.Dense(self.num_mixtures, activation=elu_plus, name='sigmas')(x)

        #mixture coefficients must sum to 1, therefore use softmax
        pis = layers.Dense(self.num_mixtures, activation='softmax', name='pis')(x)
        
        mdn_out = layers.Concatenate(name='outputs')([mus,sigmas,pis])
        return mdn_out

    def train(self, x_train, y_train, x_test, y_test, epochs, verbose=2):
        batch_hist = LossHistory()

        History = self.Net.fit(x_train, y_train,
                batch_size=self.batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_data=(x_test,y_test),
                callbacks=[batch_hist])

"""
def calc_pdf(y, mu, var):
    val = tf.subtract(y, mu)**2
    val = tf.math.exp((-1 * val)/(2 * var)) / tf.math.sqrt(2*np.pi*var)
    return val

def calc_loss(y_true, pi, mu, var):
    out = calc_pdf(y_true, mu, var)

    out = tf.multiply(out, pi)
    out = tf.reduce_sum(out, 1, keepdims = True)
    out = tf.math.log(out + 1e-10)
    return tf.reduce_mean(out)

def sample_predictions(pi_vals, mu_vals, var_vals, samples=10):
    n, k = pi_vals.shape
    out = np.zeros((n, samples))
    for i in range(n):
        for j in range(samples):
            idx = np.random.choice(range(k), p=pi_vals[i])
            out[i,j] = np.random.normal(mu_vals[i, idx], np.sqrt(var_vals[i, idx]))
    return out
"""
