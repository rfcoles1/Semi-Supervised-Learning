from Networks import *

sys.path.insert(1, '../Utils')
from datasets import *
from augment import *
from mdn_utils import *


class MDN(Network):
    def __init__(self, input_shape):
        super().__init__()
  
        self.dirpath = 'records_mdn/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

        self.input_shape = input_shape

        wandb.init(project='NoisyStudent', entity='rfcoles')

        self.config = wandb.config
        self.config.model = "MDN"
        self.config.learning_rate = 1e-4
        self.config.batch_size = 64
        self.config.dropout = 0
        self.config.fc_depth = 3
        self.config.fc_width = 256
        self.config.num_mixes = 2 
        self.config.single_mean = False
        self.config.num_classes = 196

    def compile(self):
        inp = layers.Input(self.input_shape)
        self.Net = tf.keras.models.Model(inp, self.mdn(inp))
        
        optimizer = keras.optimizers.Adam(lr=self.config.learning_rate)
        
        self.Net.compile(optimizer=optimizer,\
            loss = mdn_loss(self.config.num_mixes),\
            metrics = [
                mdn_bias(self.config.num_mixes, self.config.num_classes),
                mdn_stdev(self.config.num_mixes, self.config.num_classes),
                mdn_MAD(self.config.num_mixes, self.config.num_classes),
                mdn_outliers(self.config.num_mixes, self.config.num_classes),
                mdn_bias_MAD(self.config.num_mixes, self.config.num_classes)
                ])

        self.callbacks = []
        #self.callbacks = [WandbCallback()]

  
    def mdn(self, x):
        base_model = tf.keras.applications.ResNet50(weights=None,\
            include_top=False, input_shape=self.input_shape)
        base_model.trainable = True
        x = base_model(x, training=True)
        x = layers.GlobalAveragePooling2D()(x)

        for i in range(self.config.fc_depth-1):
            x = layers.Dense(self.config.fc_width, activation = 'relu')(x)
            x = layers.Dropout(self.config.dropout)(x)

        x = layers.Dense(self.config.fc_width, activation = 'relu')(x)

        #if only one mean is desired, output one value then expand it using an identity layer
        if self.config.single_mean:
            mus = layers.Dense(1, name='mu')(x)
            mus = layers.Dense(self.config.num_mixes, name='mus', trainable = False,\
                kernel_initializer = 'ones', bias_initializer = 'zeros')(mus)
        else:
            mus = layers.Dense(self.config.num_mixes, name='mus')(x)

        #std must be greater than 0, #try exp, softplus, or elu + 1
        sigmas = layers.Dense(self.config.num_mixes, activation=elu_plus, name='sigmas')(x)
        #mixture coefficients must sum to 1, therefore use softmax
        pis = layers.Dense(self.config.num_mixes, activation='softmax', name='pis')(x)
        
        mdn_out = layers.Concatenate(name='outputs')([mus,sigmas,pis])
        return mdn_out

    def train(self, x_train, y_train, x_test, y_test, epochs, verbose=2):
        History = self.Net.fit(x_train, y_train,
                batch_size=self.config.batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_data=(x_test,y_test),
                callbacks=self.callbacks)

        epochs_arr = np.arange(self.curr_epoch, self.curr_epoch+epochs, 1)
        iterations = np.ceil(np.shape(x_train)[0]/self.config.batch_size)

        self.hist['epochs'].append(epochs_arr)
        self.hist['iterations'].append(epochs_arr*iterations)

        self.hist['train_loss'].append(History.history['loss'])
        self.hist['train_bias'].append(History.history['bias'])
        self.hist['train_stdev'].append(History.history['stdev'])
        self.hist['train_MAD'].append(History.history['MAD'])
        self.hist['train_outliers'].append(History.history['outliers'])      
        self.hist['train_bias_MAD'].append(History.history['bias_MAD'])
        
        self.hist['test_loss'].append(History.history['val_loss'])
        self.hist['test_bias'].append(History.history['val_bias'])
        self.hist['test_stdev'].append(History.history['val_stdev'])
        self.hist['test_MAD'].append(History.history['val_MAD'])
        self.hist['test_outliers'].append(History.history['val_outliers'])
        self.hist['test_bias_MAD'].append(History.history['val_bias_MAD'])

        self.curr_epoch += epochs
        
        wandb.finish()
