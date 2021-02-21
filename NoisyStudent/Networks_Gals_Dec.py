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
        
        enc_lr = 1e-6
        dec_lr = 1e-6
        enc_optimizer = keras.optimizers.Adam(lr=enc_lr)            
        dec_optimizer = keras.optimizers.Adam(lr=dec_lr)
        
        self.inp = layers.Input(input_shape, name='ae_input')
        self.Enc = tf.keras.models.Model(self.inp, \
            self.encoder(self.inp)[0], name='encoder')
        self.Dec = tf.keras.models.Model(self.inp, \
            self.decoder(tf.concat(self.encoder(self.inp),axis=1)), name='decoder')
        
        self.Enc.compile(loss=tf.keras.losses.MSE,\
            optimizer=enc_optimizer,\
            metrics = [abs_bias_loss, MAD_loss, bias_MAD_loss])
        self.Dec.compile(loss=tf.keras.losses.MSE,\
            optimizer=dec_optimizer)

    def encoder(self,y):
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,\
            input_shape=self.input_shape)
        base_model.trainabe = True
        model_out = base_model(y, training=True)
        model_out = layers.GlobalAveragePooling2D()(model_out)
        
        x = layers.Dense(512, activation = 'relu')(model_out)
        x = layers.Dense(256, activation = 'relu')(x)
        x = layers.Dense(128, activation = 'relu')(x)
        
        x_out = layers.Dense(self.num_out)(x)
        z_out = layers.Dense(self.num_z)(x)
        
        return x_out,z_out 

    def decoder(self,z):
        #TODO this decoder was made in a rush and will be changed in future
        #These layers assume a shape (32x32x5)

        y = layers.Dense(self.num_z + self.num_out)(z)
        y = layers.Dense(256, activation = 'relu')(y)
        y = layers.Dense(512, activation = 'relu')(y)
        y = layers.Dense(8192)(y)
        y = layers.Reshape([2,2,2048])(y)

        y = layers.Conv2DTranspose(512,3)(y)
        y = layers.Conv2DTranspose(128,5)(y)
        y = layers.Conv2DTranspose(64,9)(y)
        y = layers.Conv2DTranspose(5,17)(y)

        return y


    def train(self, x_train, y_train, x_test, y_test, epochs):
        
        N = int(len(x_train)/self.batch_size)

        reglosses = np.zeros(self.checkpoint)
        reconlosses = np.zeros(self.checkpoint)

        it = 0               
        while it < epochs:    
            x_train, y_train = shuffle(x_train,y_train)
            
            for j in range(N):
                x_true = x_train[j*self.batch_size: (j+1)*self.batch_size]
                y_true = y_train[j*self.batch_size: (j+1)*self.batch_size]
       
                #Im working on another version where both are trained simulateously
                enc_loss = self.Enc.train_on_batch(x_true, y_true)
                dec_loss = self.Dec.train_on_batch(x_true, x_true)
            
                reglosses[it%self.checkpoint] = enc_loss[0]
                reconlosses[it%self.checkpoint] = dec_loss

            if it % self.checkpoint == 0:
                ##TODO add correct saving and validation tests to checkpoints
                self.curr_epoch += self.checkpoint
                
                print('Iterations %d' % self.curr_epoch)
                print('Regression Loss %f' % np.mean(reglosses))
                print('Reconstruction Loss %f' % np.mean(reconlosses))

                self.losses['epochs'].append(self.curr_epoch)
                self.losses['regloss'].append(np.mean(reglosses))
                self.losees['reconloss'].append(np.mean(reconlosses))
                
                reglosses = np.zeros(self.checkpoint)
                reconlosses = np.zeros(self.checkpoint)

                self.save()
                
            it += 1

    def predict_enc(self, x_test):
        preds = self.Enc.predict(x_test, batch_size=self.batch_size, verbose=0)
        return preds

    def predict_ae(self, x_test):
        preds = self.Dec.predict(x_test, batch_size=self.batch_size, verbose=0)
        return preds

    def save(self, path):
        f = open(self.dirpath + path + '.pickle', 'wb')
        pickle.dump(self.hist,f)
        f.close()
        self.Enc.save_weights(self.dirpath + path + '_enc.h5')
        self.Dec.save_weights(self.dirpath + path + '_dec.h5')

    def load(self, path):
        f = open(self.dirpath + path + '.pickle', 'rb')
        self.hist = pickle.load(f)
        f.close()
        self.Enc.load_weights(self.dirpath + path + '_enc.h5')
        self.Dec.load_weights(self.dirpath + path + '_dec.h5')
        self.curr_epoch = self.hist['epochs'][-1][-1]
