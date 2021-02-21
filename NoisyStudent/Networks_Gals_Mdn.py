from Networks import *

sys.path.insert(1, '../Utils')
from metrics import *
from datasets import *
from augment import *
from mdn_utils import *


class MDN(Network):
    def __init__(self, input_shape, noise=False):
        super().__init__()
  
        self.dirpath = 'records_z/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        
        self.batch_size = 64
        self.input_shape = input_shape
        self.num_out = 1
        self.num_mixtures = 5

        lr = 1e-6
        optimizer = keras.optimizers.Adam(lr=lr)            

        self.inp = layers.Input(self.input_shape)
        self.Net = tf.keras.models.Model(self.inp, self.network_mdn(self.inp))
        self.Net.compile(loss = mdn_loss(self.num_mixtures),  optimizer=optimizer)

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

        