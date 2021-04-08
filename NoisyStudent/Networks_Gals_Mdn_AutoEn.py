from Networks import *

sys.path.insert(1, '../Utils')
from metrics import *
from datasets import *
from augment import *
from mdn_utils import *

class MDN_AE(Network):
    def __init__(self, input_shape):
        super().__init__()
  
        self.dirpath = 'records_mdn_ae/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        
        self.batch_size = 64
        self.input_shape = input_shape
        self.num_out = 1
        self.num_mixtures = 1
        self.lr = 1e-4
        self.dropout = 0
        self.single_mean = False

        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss',\
                                patience=10, verbose=2, restore_best_weights=True)]

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
        
        optimizer = keras.optimizers.Adam(lr=self.lr)
        self.Net.compile(optimizer=optimizer, \
            loss={'regressor': mdn_loss(self.num_mixtures), 'decoder': tf.keras.losses.MSE})


    def encoder(self, y):
        self.base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,\
            input_shape=self.input_shape)
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
        
        y = layers.Dense(512, activation = 'relu')(y)
        y = layers.Dropout(self.dropout)(y)
        y = layers.Dense(256, activation = 'relu')(y)
        y = layers.Dropout(self.dropout)(y)
        y = layers.Dense(128, activation = 'relu')(y)
    
        if self.single_mean:
            mus = layers.Dense(1, name='mu')(y)
            mus = layers.Dense(self.num_mixtures, name='mus', trainable = False,\
                kernel_initializer = 'ones', bias_initializer = 'zeros')(mus)
        else:
            mus = layers.Dense(self.num_mixtures, name='mus')(y)
        sigmas = layers.Dense(self.num_mixtures, activation=elu_plus, name='sigmas')(y)
        pis = layers.Dense(self.num_mixtures, activation='softmax', name='pis')(y)
        
        mdn_out = layers.Concatenate(name='outputs')([mus, sigmas, pis])
        return mdn_out
    
   
    def train(self, x_train, x_train_aug, y_train, x_test, x_test_aug, y_test, epochs, verbose=2):

        History = self.Net.fit(x_train_aug, {'regressor': y_train, 'decoder': x_train},
                batch_size=self.batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_data=(x_test_aug, {'regressor': y_test, 'decoder': x_test}),
                callbacks=self.callbacks)

        epochs_arr = np.arange(self.curr_epoch, self.curr_epoch+epochs, 1)
        iterations = np.ceil(np.shape(x_train)[0]/self.batch_size)

        self.hist['epochs'].append(epochs_arr)
        self.hist['iterations'].append(epochs_arr*iterations)

        self.hist['train_regress_loss'].append(History.history['regressor_loss'])
        self.hist['train_recon_loss'].append(History.history['decoder_loss'])

        self.hist['test_regress_loss'].append(History.history['val_regressor_loss'])
        self.hist['test_recon_loss'].append(History.history['val_decoder_loss'])
        
        self.curr_epoch += epochs

    def predict(self, x_test):
        preds = self.Net.predict(x_test, batch_size=self.batch_size, verbose=0)
        return preds


    def save(self, path):
        f = open(self.dirpath + path + '.pickle', 'wb')
        pickle.dump(self.hist,f)
        f.close()
        self.Enc.save_weights(self.dirpath + path + '_enc.h5')
        self.Dec.save_weights(self.dirpath + path + '_dec.h5')
        self.Reg.save_weights(self.dirpath + path + '_reg.h5')

    def load(self, path):
        f = open(self.dirpath + path + '.pickle', 'rb')
        self.hist = pickle.load(f)
        f.close()
        self.Enc.load_weights(self.dirpath + path + '_enc.h5')
        self.Dec.load_weights(self.dirpath + path + '_dec.h5')
        self.Reg.load_weights(self.dirpath + path + '_reg.h5')
