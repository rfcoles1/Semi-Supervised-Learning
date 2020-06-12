from Networks import *

#import tensorflow_probability as tfp

def abs_bias_loss(y_true, y_pred):
    loss = tf.reduce_mean(abs(y_true - y_pred))/(1 + y_true)
    return loss 

def MAD_loss(y_true, y_pred):
    resid = (y_true - y_pred)/(1 + y_true)
    median = tf.contrib.distributions.percentile(resid,50.)
    #median = tfp.stats.percentile(resid, 50.0, interpolation='midpoint')
    return tf.reduce_mean(abs(resid - median))

def bias_MAD_loss(y_true, y_pred):
    resid = (y_true - y_pred)/(1 + y_true)
    median = tf.contrib.distributions.percentile(resid,50.)
    #median = tfp.stats.percentile(resid, 50.0, interpolation='midpoint')
    return tf.sqrt(tf.reduce_mean(abs(resid))) + 1.4826 * (tf.reduce_mean(abs(resid - median)))

class Network_z(Network):
    def __init__(self, input_shape, num_out=1, noise=False):
        super().__init__()
  
        self.dirpath = 'records_z/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

        self.batch_size = 16
        self.input_shape = input_shape
        self.num_out = num_out

        self.inp = layers.Input(input_shape)
        if noise==True:
            self.Net = tf.keras.models.Model(self.inp, self.CNN_noise(self.inp))
        else:
            self.Net = tf.keras.models.Model(self.inp, self.CNN(self.inp))
        
        self.Net.compile(loss=tf.keras.losses.MSE,
              optimizer=tf.keras.optimizers.Adadelta(learning_rate=self.lr),
              metrics = [abs_bias_loss, MAD_loss, bias_MAD_loss])
   
    def CNN(self, x):
        y = layers.Conv2D(128, (4, 4), activation=layers.LeakyReLU(alpha=0.1),\
                         input_shape=self.input_shape)(x)
        y = layers.Conv2D(256, (4, 4), activation=layers.LeakyReLU(alpha=0.1))(y)
        y = layers.MaxPooling2D(pool_size=(2,2))(y)
        y = layers.Flatten()(y)
        y = layers.Dense(1024, activation=layers.LeakyReLU(alpha=0.1))(y)
        y = layers.Dense(self.num_out)(y)
        return y
    
    def CNN_noise(self, x):
        y = layers.Conv2D(128, (4, 4), activation=layers.LeakyReLU(alpha=0.1),\
                         input_shape=self.input_shape)(x)
        y = layers.Conv2D(256, (4, 4), activation=layers.LeakyReLU(alpha=0.1))(y)
        y = layers.MaxPooling2D(pool_size=(2,2))(y)
        y = layers.Dropout(0.4)(y)
        y = layers.Flatten()(y)
        y = layers.Dense(1024, activation=layers.LeakyReLU(alpha=0.1))(y)
        y = layers.Dropout(0.4)(y)
        y = layers.Dense(self.num_out)(y)
        return y


    def train(self, x_train, y_train, x_test, y_test, epochs, verbose=2):
     
        history = self.Net.fit(x_train, y_train, 
                 batch_size=self.batch_size, 
                 epochs=epochs, 
                 verbose=verbose,
                 validation_data=(x_test, y_test))
        
        epochs = np.arange(self.curr_epoch, self.curr_epoch+epochs, 1))
        iterations = np.ceil(np.shape(x_train)[0]/self.batch_size)
        
        self.hist['epochs'].append(epochs)
        self.hist['iterations'].append(epochs*iterations)
        self.hist['train_mse'].append(history.history['loss'])
        self.hist['train_abs_bias'].append(history.history['abs_bias_loss'])
        self.hist['train_MAD_loss'].append(history.history['MAD_loss'])
        self.hist['train_bias_MAD_loss'].append(history.history['bias_MAD_loss'])
        self.hist['test_MSE'].append(history.history['val_loss'])
        self.hist['test_abs_bias'].append(history.history['val_abs_bias_loss'])
        self.hist['test_MAD_loss'].append(history.history['val_MAD_loss'])
        self.hist['test_bias_MAD_loss'].append(history.history['val_bias_MAD_loss'])
        
        self.curr_epoch += epochs
