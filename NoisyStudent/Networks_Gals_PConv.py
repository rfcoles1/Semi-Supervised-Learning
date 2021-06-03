from Networks import *

sys.path.insert(1, '../Utils')
from metrics import *
from datasets import *
from augment import *
from pconv import PConv_ResNet50

class Regressor(Network):
    def __init__(self, input_shape):
        super().__init__()
  
        self.dirpath = 'records_regress/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        
        self.input_shape = input_shape
       
        wandb.init(project='NoisyStudent', entity='rfcoles')
         
        self.config = wandb.config
        self.config.model = "Regressor"
        self.config.learning_rate = 1e-4
        self.config.batch_size = 64
        self.config.dropout = 0
        self.config.encoder = "PConv_Resnet50"
        self.config.fc_depth = 3
        self.config.fc_width = 256

    def compile(self):
        inp = layers.Input(self.input_shape)
        msk = layers.Input(self.input_shape)
        self.Net = tf.keras.models.Model([inp,msk], self.regressor(inp, msk))

        optimizer = keras.optimizers.Adam(lr=self.config.learning_rate)          
        
        self.Net.compile(optimizer = optimizer,\
            loss=tf.keras.losses.MSE,\
            metrics = [bias, stdev, MAD, outliers, bias_MAD])

        self.callbacks = []
        #self.callbacks = [WandbCallback()]


    def regressor(self, x, msk):
        base_model = PConv_ResNet50(input_shape=self.input_shape)
        base_model.trainable = True
        x = base_model([x,msk])
        x = layers.GlobalAveragePooling2D()(x)

        for i in range(self.config.fc_depth-1):
            x = layers.Dense(self.config.fc_width, activation = 'relu')(x)
            x = layers.Dropout(self.config.dropout)(x)
        
        x = layers.Dense(self.config.fc_width, activation = 'relu')(x)
        x = layers.Dense(1)(x)
        return x
    
    def train(self, x_train, x_train_masks, y_train,\
                x_test, x_test_masks, y_test, epochs, verbose=2):
        History = self.Net.fit([x_train,x_train_masks], y_train, 
                batch_size=self.config.batch_size, 
                epochs=epochs, 
                verbose=verbose,
                validation_data=([x_test,x_test_masks], y_test),
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
