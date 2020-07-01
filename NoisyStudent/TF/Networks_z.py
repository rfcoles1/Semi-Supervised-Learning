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
        
        self.lr = 1e-5
        self.batch_size = 64
        self.input_shape = input_shape
        self.num_out = num_out

        self.inp = layers.Input(input_shape)
        """
        if noise==True:
            self.Net = tf.keras.models.Model(self.inp, self.CNN_noise(self.inp))
        else:
            self.Net = tf.keras.models.Model(self.inp, self.CNN(self.inp))
        """
        self.Net = tf.keras.models.Model(self.inp, self.resnet(self.inp))

        self.Net.compile(loss=tf.keras.losses.MSE,
              optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
              metrics = [abs_bias_loss, MAD_loss, bias_MAD_loss])
  
    def resnet(self,x):
        x = layers.Conv2D(32, kernel_size=(3,3), padding='same')(x)
        x = _add_common_layers(x)

        x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)
        for i in range(3):
            project_shortcut = True if i==0 else False
            x = _residual_block(x, 64,128, project_shortcut=project_shortcut)
        
        for i in range(4):
            strides = (2,2) if i==0 else (1,1)
            x = _residual_block(x, 128,256, strides=strides, project_shortcut=project_shortcut)

        for i in range(6):
            strides = (2,2) if i==0 else (1,1)
            x = _residual_block(x, 256,512, strides=strides, project_shortcut=project_shortcut)

        x = layers.MaxPooling2D(pool_size=(2,2), padding='same')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(1024, activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dense(1)(x)
       
        return x

    def CNN(self, x):
       
        y = layers.Conv2D(32, (3, 3), activation=layers.LeakyReLU(alpha=0.1),\
                         input_shape=self.input_shape)(x)
        y = layers.Conv2D(64, (3, 3), activation=layers.LeakyReLU(alpha=0.1))(y)
        y = layers.Conv2D(128, (3, 3), activation=layers.LeakyReLU(alpha=0.1))(y)
        y = layers.Conv2D(256, (3, 3), activation=layers.LeakyReLU(alpha=0.1))(y)
        y = layers.Conv2D(512, (3, 3), activation=layers.LeakyReLU(alpha=0.1))(y)
        y = layers.MaxPooling2D(pool_size=(2,2))(y)
        y = layers.Flatten()(y)
        y = layers.Dense(1024, activation=layers.LeakyReLU(alpha=0.1))(y)
        y = layers.Dense(self.num_out)(y)
        return y
    
    def train(self, x_train, y_train, x_test, y_test, epochs, verbose=2):
     
        history = self.Net.fit(x_train, y_train, 
                 batch_size=self.batch_size, 
                 epochs=epochs, 
                 verbose=verbose,
                 validation_data=(x_test, y_test))
        
        epochs_arr = np.arange(self.curr_epoch, self.curr_epoch+epochs, 1)
        iterations = np.ceil(np.shape(x_train)[0]/self.batch_size)
        
        self.hist['epochs'].append(epochs_arr)
        self.hist['iterations'].append(epochs_arr*iterations)
        self.hist['train_mse'].append(history.history['loss'])
        self.hist['train_abs_bias'].append(history.history['abs_bias_loss'])
        self.hist['train_MAD_loss'].append(history.history['MAD_loss'])
        self.hist['train_bias_MAD_loss'].append(history.history['bias_MAD_loss'])
        self.hist['test_MSE'].append(history.history['val_loss'])
        self.hist['test_abs_bias'].append(history.history['val_abs_bias_loss'])
        self.hist['test_MAD_loss'].append(history.history['val_MAD_loss'])
        self.hist['test_bias_MAD_loss'].append(history.history['val_bias_MAD_loss'])
        
        self.curr_epoch += epochs


def _add_common_layers(y):
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=0.1)(y)
    return y

def _grouped_conv(y, n_channels, strides):
    #if cardinality ==1:
    return layers.Conv2D(n_channels, kernel_size=(3,3), strides=strides, padding='same')(y)

    """
    assert not n_channels % cardinality
    _d = n_channels // cardinality

    groups = []
    for j in range(cardinality):
        group = layers.Lambda(lambda z: z[:,:,:, j* _d:j * _d + _d])(y)
        groups.append(layers.Conv2D(_d, kernel_size=(3,3), strides=strides, padding='same')(group))

    y = layers.concatenate(groups)
    return y
    """
def _residual_block(y, n_channels_in, n_channels_out, strides=(1,1), project_shortcut=False):
    shortcut = y
    y = layers.Conv2D(n_channels_in, kernel_size=(1,1), strides=(1,1), padding='same')(y)
    y = _add_common_layers(y)

    y = _grouped_conv(y, n_channels_in, strides=strides)
    y = _add_common_layers(y)
        
    y = layers.Conv2D(n_channels_out, kernel_size=(1,1), strides=(1,1), padding='same')(y)
    y = layers.BatchNormalization()(y)
        
    if project_shortcut or strides!=(1,1):
        shortcut = layers.Conv2D(n_channels_out, kernel_size=(1,1),\
            strides=strides, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.add([shortcut,y])
    y = layers.LeakyReLU(alpha=0.1)(y)
    return y
    


