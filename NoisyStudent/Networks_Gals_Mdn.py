from Networks import *

sys.path.insert(1, '../Utils')
from metrics import *
from datasets import *
from augment import *
from mdn_utils import *


class MDN(Network):
    def __init__(self, input_shape):
        super().__init__()
  
        self.dirpath = 'records_mdn/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        
        self.batch_size = 64
        self.input_shape = input_shape
        self.num_mixtures = 2 
        self.lr = 1e-4
        self.dropout = 0
        self.patience = 25
        self.single_mean = False

    def compile(self):
        inp = layers.Input(self.input_shape)
        self.Net = tf.keras.models.Model(inp, self.mdn(inp))

        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss',\
                                patience=self.patience, verbose=2, restore_best_weights=True)]

        optimizer = keras.optimizers.Adam(lr=self.lr)            
        self.Net.compile(optimizer=optimizer, \
            loss = mdn_loss(self.num_mixtures))
   
   
    def mdn(self, x):
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,\
            input_shape=self.input_shape)
        base_model.trainable = True
        x = base_model(x, training=True)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(512, activation = 'relu')(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(256, activation = 'relu')(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(128, activation = 'relu', name='latent')(x)
        
        #if only one mean is desired, output one value then expand it using an identity layer
        if self.single_mean:
            mus = layers.Dense(1, name='mu')(x)
            mus = layers.Dense(self.num_mixtures, name='mus', trainable = False,\
                kernel_initializer = 'ones', bias_initializer = 'zeros')(mus)
        else:
            mus = layers.Dense(self.num_mixtures, name='mus')(x)

        #std must be greater than 0, #try exp, softplus, or elu + 1
        sigmas = layers.Dense(self.num_mixtures, activation=elu_plus, name='sigmas')(x)
        #mixture coefficients must sum to 1, therefore use softmax
        pis = layers.Dense(self.num_mixtures, activation='softmax', name='pis')(x)
        
        mdn_out = layers.Concatenate(name='outputs')([mus,sigmas,pis])
        return mdn_out

    def train(self, x_train, y_train, x_test, y_test, epochs, verbose=2):

        History = self.Net.fit(x_train, y_train,
                batch_size=self.batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_data=(x_test,y_test),
                callbacks=self.callbacks)

        epochs_arr = np.arange(self.curr_epoch, self.curr_epoch+epochs, 1)
        iterations = np.ceil(np.shape(x_train)[0]/self.batch_size)

        self.hist['epochs'].append(epochs_arr)
        self.hist['iterations'].append(epochs_arr*iterations)
        self.hist['train_loss'].append(History.history['loss'])
        self.hist['test_loss'].append(History.history['val_loss'])

        self.curr_epoch += epochs
