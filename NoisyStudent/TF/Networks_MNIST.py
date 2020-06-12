from Networks import *

class Network_MNIST(Network):
    def __init__(self, input_shape, num_out, noise=False):
        super().__init__()

        self.dirpath = 'records_MNIST/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)

        self.batch_size = 128
        self.input_shape = input_shape
        self.num_out = num_out

        self.inp = layers.Input(input_shape)
        if noise==True:
            self.Net = tf.keras.models.Model(self.inp, self.CNN_noise(self.inp))
        else:
            self.Net = tf.keras.models.Model(self.inp, self.CNN(self.inp))
        
        self.Net.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(learning_rate=self.lr),
              metrics=['accuracy'])
   
    def CNN(self, x):
        y = layers.Conv2D(32, (3, 3), activation='relu',\
                         input_shape=self.input_shape)(x)
        y = layers.Conv2D(64, (3, 3), activation='relu')(y)
        y = layers.MaxPooling2D(pool_size=(2, 2))(y)
        y = layers.Flatten()(y)
        y = layers.Dense(128, activation='relu')(y)
        y = layers.Dense(self.num_out, activation='softmax')(y)
        return y
    
    def CNN_noise(self, x):
        y = layers.Conv2D(32, (3, 3), activation='relu',\
                         input_shape=self.input_shape)(x)
        y = layers.Conv2D(64, (3, 3), activation='relu')(y)
        y = layers.MaxPooling2D(pool_size=(2, 2))(y)
        y = layers.Dropout(0.5)(y)
        y = layers.Flatten()(y)
        y = layers.Dense(128, activation='relu')(y)
        y = layers.Dropout(0.5)(y)
        y = layers.Dense(self.num_out, activation='softmax')(y)
        return y


    def train(self, x_train, y_train, x_test, y_test, epochs, verbose=1):
     
        history = self.Net.fit(x_train, y_train, 
                 batch_size=self.batch_size, 
                 epochs=epochs, 
                 verbose=verbose,
                 validation_data=(x_test, y_test))
    
        epochs_arr = np.arange(self.curr_epoch, self.curr_epoch+epochs,1)
        iterations = np.ceil(np.shape(x_train)[0]/self.batch_size)

        self.hist['epochs'].append(epochs_arr)
        self.hist['iterations'].append(epochs_arr*iterations)
        self.hist['train_loss'].append(history.history['loss'])
        self.hist['test_loss'].append(history.history['val_loss'])
        try:
            self.hist['train_acc'].append(history.history['accuracy'])
            self.hist['test_acc'].append(history.history['val_accuracy'])
        except:
            self.hist['train_acc'].append(history.history['acc'])
            self.hist['test_acc'].append(history.history['val_acc'])
        
        self.curr_epoch += epochs
        
