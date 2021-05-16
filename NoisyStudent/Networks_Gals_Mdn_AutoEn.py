from Networks import *

sys.path.insert(1, '../Utils')
from datasets import *
from augment import *
from mdn_utils import *

class MDN_AE(AutoEncoder):
    def __init__(self, input_shape):
        super().__init__()
  
        self.dirpath = 'records_mdn_ae/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

        self.input_shape = input_shape

        run = wandb.init(project='NoisyStudent_AutoEnc', entity='rfcoles')

        self.config = wandb.config
        self.config.model = "AutoEnc_MDN"
        self.config.learning_rate = 1e-4
        self.config.batch_size = 64
        self.config.dropout = 0
        self.config.fc_depth = 3
        self.config.fc_width = 256
        self.config.num_mixes = 2
        self.config.single_mean = False
        self.config.num_classes = 196

    def compile(self):
        Enc_inp = layers.Input(self.input_shape, name='encoder_input')
        self.Enc = tf.keras.models.Model(Enc_inp, \
            self.encoder(Enc_inp), name='encoder')
       
        Dec_inp = layers.Input(shape=(1,1,2048), name='decoder_input')
        self.Dec = tf.keras.models.Model(Dec_inp, \
            self.decoder(Dec_inp), name='decoder')
        
        Reg_inp = layers.Input(shape=(1,1,2048), name='regressor_input')
        self.Reg = tf.keras.models.Model(Reg_inp, \
            self.regressor(Reg_inp), name='regressor')

        outputs = [self.Dec(self.Enc(Enc_inp)), self.Reg(self.Enc(Enc_inp))]
        self.Net = tf.keras.models.Model(inputs=Enc_inp,\
            outputs=outputs)
       
        optimizer = keras.optimizers.Adam(lr=self.config.learning_rate)
        self.Net.compile(optimizer=optimizer, \
            loss={'regressor': mdn_loss(self.config.num_mixes), 'decoder': tf.keras.losses.MSE},\
            loss_weights=[1,1],
            metrics = {'regressor': [
                mdn_bias(self.config.num_mixes, self.config.num_classes),
                mdn_stdev(self.config.num_mixes, self.config.num_classes),
                mdn_MAD(self.config.num_mixes, self.config.num_classes),
                mdn_outliers(self.config.num_mixes, self.config.num_classes),
                mdn_bias_MAD(self.config.num_mixes, self.config.num_classes)
                ]})

        self.callbacks = []
        #self.callbacks = [WandbCallback()]

    def encoder(self, y):
        self.base_model = tf.keras.applications.ResNet50(weights=None,\
            input_shape=self.input_shape, include_top=False)
        self.base_model.trainable = True
        z = self.base_model(y, training=True)
        return z

    def decoder(self, z):
        x = layers.Conv2DTranspose(512, 4)(z)
        x = layers.Conv2DTranspose(128, 5)(x)
        x = layers.Conv2DTranspose(64, 9)(x)
        x = layers.Conv2DTranspose(5, 17)(x)
        return x

    def regressor(self,z):
        y = layers.Flatten()(z)
        
        for i in range(self.config.fc_depth-1):
            y = layers.Dense(self.config.fc_width, activation = 'relu')(y)
            y = layers.Dropout(self.config.dropout)(y)

        y = layers.Dense(self.config.fc_width, activation = 'relu')(y)
    
        if self.config.single_mean:
            mus = layers.Dense(1, name='mu')(y)
            mus = layers.Dense(self.config.num_mixes, name='mus', trainable = False,\
                kernel_initializer = 'ones', bias_initializer = 'zeros')(mus)
        else:
            mus = layers.Dense(self.config.num_mixes, name='mus')(y)
        sigmas = layers.Dense(self.config.num_mixes, activation=elu_plus, name='sigmas')(y)
        pis = layers.Dense(self.config.num_mixes, activation='softmax', name='pis')(y)
        
        mdn_out = layers.Concatenate(name='outputs')([mus, sigmas, pis])
        return mdn_out
    
   
    def train(self, x_train, x_train_aug, y_train, x_test, x_test_aug, y_test, epochs, verbose=2):

        History = self.Net.fit(x_train_aug, {'regressor': y_train, 'decoder': x_train},
                batch_size=self.config.batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_data=(x_test_aug, {'regressor': y_test, 'decoder': x_test}),
                callbacks=self.callbacks)

        epochs_arr = np.arange(self.curr_epoch, self.curr_epoch+epochs, 1)
        iterations = np.ceil(np.shape(x_train)[0]/self.config.batch_size)

        self.hist['epochs'].append(epochs_arr)
        self.hist['iterations'].append(epochs_arr*iterations)

        self.hist['train_loss'].append(History.history['regressor_loss'])
        self.hist['train_bias'].append(History.history['regressor_bias'])
        self.hist['train_stdev'].append(History.history['regressor_stdev'])
        self.hist['train_MAD'].append(History.history['regressor_MAD'])
        self.hist['train_outliers'].append(History.history['regressor_outliers'])
        self.hist['train_bias_MAD'].append(History.history['regressor_bias_MAD'])
        self.hist['train_recon'].append(History.history['decoder_loss'])

        self.hist['test_loss'].append(History.history['val_regressor_loss'])
        self.hist['test_bias'].append(History.history['val_regressor_bias'])
        self.hist['test_stdev'].append(History.history['val_regressor_stdev'])
        self.hist['test_MAD'].append(History.history['val_regressor_MAD'])
        self.hist['test_outliers'].append(History.history['val_regressor_outliers'])
        self.hist['test_bias_MAD'].append(History.history['val_regressor_bias_MAD'])
        self.hist['test_recon'].append(History.history['val_decoder_loss'])
        
        self.curr_epoch += epochs

        wandb.finish()
