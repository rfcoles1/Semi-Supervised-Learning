from Networks import *

import tensorflow_probability as tfp

class LossHistory(keras.callbacks.Callback):
    
    """
    def __init__(self, Net, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test
    """
    def on_train_begin(self, logs={}):
        self.history = {'loss':[], 'val_loss':[]}

    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))
        #val_loss = self.Net.evaluate(self.x_test,self.y_test,verbose=0)
        #self.history['val_loss'].append(val_loss)
        

def abs_bias_loss(y_true, y_pred):
    loss = tf.reduce_mean(abs(y_true - y_pred))/(1 + y_true)
    return loss 

def MAD_loss(y_true, y_pred):
    resid = (y_true - y_pred)/(1 + y_true)
    #median = tf.contrib.distributions.percentile(resid,50.)
    median = tfp.stats.percentile(resid, 50.0, interpolation='midpoint')
    return tf.reduce_mean(abs(resid - median))

def bias_MAD_loss(y_true, y_pred):
    resid = (y_true - y_pred)/(1 + y_true)
    #median = tf.contrib.distributions.percentile(resid,50.)
    median = tfp.stats.percentile(resid, 50.0, interpolation='midpoint')
    return tf.sqrt(tf.reduce_mean(abs(resid))) + 1.4826 * (tf.reduce_mean(abs(resid - median)))

class Network_z(Network):
    def __init__(self, input_shape, noise=False):
        super().__init__()
  
        self.dirpath = 'records_z/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        
        self.batch_size = 64
        self.input_shape = input_shape
        self.num_out = 1

        #lr_decayed_fn = keras.optimizers.schedules.ExponentialDecay(\
        #    initial_learning_rate=0.01, decay_steps=1000)
        optimizer = keras.optimizers.Adam(lr=0.0001)            

        self.inp = layers.Input(input_shape)
        if noise==True:
            self.Net = tf.keras.models.Model(self.inp, self.tf_resnet_noise(self.inp))
        else:
            self.Net = tf.keras.models.Model(self.inp, self.resnet(self.inp))

        self.Net.compile(loss=tf.keras.losses.MSE,
            optimizer=optimizer,
            metrics = [abs_bias_loss, MAD_loss, bias_MAD_loss])
        self.es = tf.keras.callbacks.EarlyStopping(monitor='loss',\
            patience=10, verbose=2, restore_best_weights=True)
   
    def tf_resnet(self,x):
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,\
            input_shape=self.input_shape)
        base_model.trainabe = True
        inputs = layers.Input((32,32,5))
        h = base_model(x, training=True)
        h = layers.GlobalAveragePooling2D()(h)

        projection_1 = layers.Dense(256,activation=layers.LeakyReLU(alpha=0.1))(h)
        projection_2 = layers.Dense(128,activation=layers.LeakyReLU(alpha=0.1))(projection_1)
        projection_3 = layers.Dense(1)(projection_2)

        return projection_3
    
    def tf_resnet_noise(self,x):
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,\
            input_shape=self.input_shape)
        base_model.trainabe = True
        inputs = layers.Input((32,32,5))
        h = base_model(x, training=True)
        h = layers.GlobalAveragePooling2D()(h)

        projection_1 = layers.Dense(256,activation=layers.LeakyReLU(alpha=0.1))(h)
        projection_1 = layers.Dropout(0.5)(projection_1)
        projection_2 = layers.Dense(128,activation=layers.LeakyReLU(alpha=0.1))(projection_1)
        projection_2 = layers.Dropout(0.5)(projection_2)
        projection_3 = layers.Dense(1)(projection_2)
        
        return projection_3

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
        
        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dense(64, activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dense(1)(x)
      
        return x

    def resnet_noise(self,x):
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
        
        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1)(x)
      
        return x


    def train(self, x_train, y_train, x_test, y_test, epochs, verbose=2):
        batch_hist = LossHistory()
        
        History = self.Net.fit(x_train, y_train, 
                 batch_size=self.batch_size, 
                 epochs=epochs, 
                 verbose=verbose,
                 validation_data=(x_test, y_test),
                 callbacks=[batch_hist, self.es])
        
        epochs_arr = np.arange(self.curr_epoch, self.curr_epoch+epochs, 1)
        iterations = np.ceil(np.shape(x_train)[0]/self.batch_size)
       
        self.hist['epochs'].append(epochs_arr)
        self.hist['iterations'].append(epochs_arr*iterations)
        self.hist['train_MSE'].append(History.history['loss'])
        self.hist['batch_MSE'].append(batch_hist.history['loss'])
        self.hist['train_abs_bias'].append(History.history['abs_bias_loss'])
        self.hist['train_MAD_loss'].append(History.history['MAD_loss'])
        self.hist['train_bias_MAD_loss'].append(History.history['bias_MAD_loss'])
        self.hist['test_MSE'].append(History.history['val_loss'])
        self.hist['test_abs_bias'].append(History.history['val_abs_bias_loss'])
        self.hist['test_MAD_loss'].append(History.history['val_MAD_loss'])
        self.hist['test_bias_MAD_loss'].append(History.history['val_bias_MAD_loss'])
        
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
    


