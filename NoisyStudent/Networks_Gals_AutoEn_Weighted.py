from Networks import *

sys.path.insert(1, '../Utils')
from metrics import *
from datasets import *
from augment import *
from ae_utils import *

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

class Regressor_AE(Network):
    def __init__(self, input_shape):
        super().__init__()
  
        self.dirpath = 'records_regress_ae_weighted/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        
        self.batch_size = 64
        self.input_shape = input_shape
        self.lr = 1e-4
        self.dropout = 0
        self.patience = 25

    def compile(self):
        Enc_inp = layers.Input(self.input_shape, name='encoder_input')
        Weight_inp = layers.Input(self.input_shape, name='weight_input')
        self.Enc = tf.keras.models.Model(Enc_inp, \
            self.encoder(Enc_inp), name='encoder')
       
        #need to not hardcode the latent size
        Dec_inp = layers.Input(shape=(1,1,2048), name='decoder_input')
        self.Dec = tf.keras.models.Model(Dec_inp, \
            self.decoder(Dec_inp), name='decoder')

        Reg_inp = layers.Input(shape=(1,1,2048), name='regressor_input')
        self.Reg = tf.keras.models.Model(Reg_inp, \
            self.regressor(Reg_inp), name='regressor')

        outputs = [self.Dec(self.Enc(Enc_inp)), self.Reg(self.Enc(Enc_inp))]
        self.Net = tf.keras.models.Model(inputs=[Enc_inp, Weight_inp],\
            outputs=outputs)
       
        self.callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss',\
                                patience=self.patience, verbose=2, restore_best_weights=True)]

        optimizer = keras.optimizers.Adam(lr=self.lr)
        self.Net.compile(optimizer=optimizer, \
                loss={'regressor': tf.keras.losses.MSE, 'decoder': weighted_recon_loss(Weight_inp)},\
                loss_weights=[1,1],\
                metrics={'regressor': [abs_bias_loss, MAD_loss, bias_MAD_loss]})
        

    def encoder(self, y):
        self.base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,\
            input_shape=self.input_shape)
        self.base_model.trainabe = True
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

        y = layers.Dense(1)(y)
        return y
    
    
    def train(self, x_train, train_weights, y_train, \
            x_test, test_weights, y_test, epochs, verbose=2):

        History = self.Net.fit([x_train, train_weights], {'regressor': y_train, 'decoder': x_train},
            batch_size=self.batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=([x_test, test_weights], {'regressor': y_test, 'decoder': x_test}),
            callbacks=self.callbacks)

        epochs_arr = np.arange(self.curr_epoch, self.curr_epoch+epochs, 1)
        iterations = np.ceil(np.shape(x_train)[0]/self.batch_size)

        self.hist['epochs'].append(epochs_arr)
        self.hist['iterations'].append(epochs_arr*iterations)
        
        self.hist['train_MSE'].append(History.history['regressor_loss'])
        self.hist['train_abs_bias'].append(History.history['regressor_abs_bias_loss'])
        self.hist['train_MAD_bias'].append(History.history['regressor_MAD_loss'])
        self.hist['train_bias_MAD_loss'].append(History.history['regressor_bias_MAD_loss'])
        self.hist['train_recon_loss'].append(History.history['decoder_loss'])
        
        self.hist['test_MSE'].append(History.history['val_regressor_loss'])
        self.hist['test_abs_bias'].append(History.history['val_regressor_abs_bias_loss'])
        self.hist['test_MAD_bias'].append(History.history['val_regressor_MAD_loss'])
        self.hist['test_bias_MAD_loss'].append(History.history['val_regressor_bias_MAD_loss'])
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
