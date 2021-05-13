from Networks import *

sys.path.insert(1, '../Utils')
from metrics import *
from datasets import *
from augment import *
        
class Regressor_AE(AutoEncoder):
    def __init__(self, input_shape):
        super().__init__()
  
        self.dirpath = 'records_regress_ae/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
       
        self.input_shape = input_shape

        run = wandb.init(project='NoisyStudent', entity='rfcoles')

        self.config = wandb.config
        self.config.model = 'AutoEnc_Regressor'
        self.config.learning_rate = 1e-4
        self.config.batch_size = 64
        self.config.dropout = 0

    def compile(self):
        Enc_inp = layers.Input(self.input_shape, name='encoder_input')
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
        self.Net = tf.keras.models.Model(inputs=Enc_inp,\
            outputs=outputs)

        self.callbacks = [WandbCallback()]
            #[tf.keras.callbacks.EarlyStopping(monitor='val_loss',\
                #patience=self.patience, verbose=2, restore_best_weights=True)]

        optimizer = keras.optimizers.Adam(lr=self.config.learning_rate)
        self.Net.compile(optimizer=optimizer, \
            loss={'regressor': tf.keras.losses.MSE, 'decoder': tf.keras.losses.MSE},\
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

    def regressor(self, z):
        y = layers.Flatten()(z)

        y = layers.Dense(512, activation = 'relu')(y)
        y = layers.Dropout(self.config.dropout)(y)
        y = layers.Dense(256, activation = 'relu')(y)
        y = layers.Dropout(self.config.dropout)(y)
        y = layers.Dense(128, activation = 'relu')(y)
        
        y = layers.Dense(1)(y)
        return y
    
    
    def train(self, x_train, x_train_aug, y_train, \
            x_test, x_test_aug, y_test, epochs, verbose=2):

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

