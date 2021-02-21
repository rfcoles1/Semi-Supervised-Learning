from Networks import *

sys.path.insert(1, '../Utils')
from metrics import *
from datasets import *
from augment import *
        
class AutoEnc(Network):
    def __init__(self, input_shape, noise=False):
        super().__init__()
  
        self.dirpath = 'records_z/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        
        self.batch_size = 64
        self.input_shape = input_shape
        self.num_out = 1
        self.num_z = 128
        self.checkpoint = 1

        lr = 1e-6
        optimizer = keras.optimizers.Adam(lr=lr)            

        Enc_inp = layers.Input(input_shape, name='encoder_input')
        self.Enc = tf.keras.models.Model(Enc_inp, \
            self.encoder(Enc_inp), name='encoder')
       
        Dec_inp = layers.Input(shape=(self.num_z), name='decoder_input')
        self.Dec = tf.keras.models.Model(Dec_inp, \
            self.decoder(Dec_inp), name='decoder')
        
        Reg_inp = layers.Input(shape=(self.num_z), name='regressor_input')
        self.Reg = tf.keras.models.Model(Reg_inp, \
            self.regressor(Reg_inp), name='regressor')

        outputs = [self.Dec(self.Enc(Enc_inp)), self.Reg(self.Enc(Enc_inp))]
        self.Net = tf.keras.models.Model(inputs=Enc_inp,\
            outputs=outputs)
        
        #recon_loss = tf.keras.losses.MSE(Enc_inp, outputs[0])
        #self.Net.add_loss(recon_loss)
        
        self.Net.compile(optimizer=optimizer, \
                loss={'regressor': tf.keras.losses.MSE, 'decoder': tf.keras.losses.MSE},\
                metrics={'regressor': [abs_bias_loss, MAD_loss, bias_MAD_loss]})


    def encoder(self,y):
        self.base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,\
            input_shape=self.input_shape)
        self.base_model.trainabe = True
        model_out = self.base_model(y, training=True)
        model_out = layers.GlobalAveragePooling2D()(model_out)
        
        x = layers.Dense(512, activation = 'relu')(model_out)
        x = layers.Dense(256, activation = 'relu')(x)
        x = layers.Dense(128, activation = 'relu')(x)
        
        z_out = layers.Dense(self.num_z)(x)
        return z_out


    def decoder(self,x):
        y = layers.Dense(self.num_z)(x)
        y = layers.Dense(256, activation = 'relu')(y)
        y = layers.Dense(512, activation = 'relu')(y)
        y = layers.Dense(8192)(y)
        y = layers.Reshape([2,2,2048])(y)
        
        y = layers.Conv2DTranspose(512,3)(y)
        y = layers.Conv2DTranspose(128,5)(y)
        y = layers.Conv2DTranspose(64,9)(y)
        y = layers.Conv2DTranspose(5,17)(y)
        
        return y

    def regressor(self,x):
        y = layers.Dense(128)(x)
        y = layers.Dense(1)(x)
        return y
    
    
    def train(self, x_train, x_train_aug, y_train, x_test, x_test_aug, y_test, epochs, verbose=2):
        batch_hist = LossHistory()

        History = self.Net.fit(x_train_aug, {'regressor': y_train, 'decoder': x_train},
            batch_size=self.batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(x_test_aug, {'regressor': y_test, 'decoder': x_test}),
            callbacks=[batch_hist])

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
