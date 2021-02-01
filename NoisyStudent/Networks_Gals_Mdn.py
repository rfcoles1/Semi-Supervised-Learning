
from Networks import *

sys.path.insert(1, '../Utils')
from metrics import *
from datasets import *
from augment import *
        
class MDN(Network):
    def __init__(self, input_shape, noise=False):
        super().__init__()
  
        self.dirpath = 'records_z/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        
        self.batch_size = 64
        self.input_shape = input_shape
        self.num_out = 1
        self.num_mixtures = 20
        self.lr = 0.0001
        self.checkpoint = 1
        self.optimizer = keras.optimizers.Adam(lr=self.lr)            

        self.inp = layers.Input(self.input_shape)
        self.mdn = tf.keras.models.Model(self.inp, self.tf_resnet_mdn(self.inp))

   
    def tf_resnet_mdn(self,x):
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,\
            input_shape=self.input_shape)
        base_model.trainable = True
        x = base_model(x, training=True)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(256,activation='relu')(x)
        x = layers.Dense(128,activation='relu')(x)
        mu = layers.Dense(self.num_mixtures)(x)

        #std must be greater than 0,
        #try exp, softplus, or elu + 1
        var = layers.Dense(self.num_mixtures, activation='softplus')(x)

        #mixture coefficients must sum to 1, therefore use softmax
        pi = layers.Dense(self.num_mixtures, activation='softmax')(x)

        return [mu,var,pi]

    def train_step(self, x_train, y_train):
        with tf.GradientTape() as tape:
            mu_, var_, pi_ = self.mdn(x_train, training=True)
            loss = calc_loss(y_train, pi_, mu_, var_)
        gradients = tape.gradient(loss, self.mdn.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.mdn.trainable_variables))
        return loss

    def train(self, dataset, epochs):
        for i in range(epochs):
            for x_train, y_train in dataset:
                loss = self.train_step(x_train, y_train)
            
            if i % self.checkpoint == 0:
                print(loss)

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
