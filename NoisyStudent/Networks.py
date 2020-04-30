import numpy as np
import pickle
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as layers
from tensorflow.keras import backend as K

dirpath = './records/'

class Network():
    def __init__(self, input_shape, num_class, noise=False):
        self.batch_size = 128
        self.input_shape = input_shape
        self.num_classes = num_class

        self.inp = layers.Input(input_shape)
        if noise==True:
            self.Net = tf.keras.models.Model(self.inp, self.CNN_noise(self.inp))
        else:
            self.Net = tf.keras.models.Model(self.inp, self.CNN(self.inp))
        
        self.Net.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])
   
        self.hist = defaultdict(list)
        self.curr_epoch = 0

    def CNN(self, x):
        y = layers.Conv2D(32, (3, 3), activation='relu',\
                         input_shape=self.input_shape)(x)
        y = layers.Conv2D(64, (3, 3), activation='relu')(y)
        y = layers.MaxPooling2D(pool_size=(2, 2))(y)
        y = layers.Flatten()(y)
        y = layers.Dense(128, activation='relu')(y)
        y = layers.Dense(self.num_classes, activation='softmax')(y)
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
        y = layers.Dense(self.num_classes, activation='softmax')(y)
        return y


    def train(self, x_train, y_train, x_test, y_test, epochs, verbose=1):
     
        history = self.Net.fit(x_train, y_train, 
                 batch_size=self.batch_size, 
                 epochs=epochs, 
                 verbose=verbose,
                 validation_data=(x_test, y_test))
    
        self.hist['iterations'].append(np.arange(self.curr_epoch,self.curr_epoch+epochs,verbose))
        self.hist['train_loss'].append(history.history['loss'])
        self.hist['train_acc'].append(history.history['acc'])
        self.hist['test_loss'].append(history.history['val_loss'])
        self.hist['test_acc'].append(history.history['val_acc'])

        self.curr_epoch += epochs

    def predict(self, x_test):
        preds = self.Net.predict(x_test,
                        batch_size=self.batch_size,
                        verbose=0) 
        return preds
   
    def evaluate(self, x_test, y_test):
        return self.Net.evaluate(x_test,y_test)

    def history(self):
        return self.hist

    #def save(self, path):
    #    self.net.save(dirpath + path + '.h5')

    #def load(self, path):
    #    self.Net = tf.keras.models.load_model(dirpath + path + '.h5')

    def save(self, path):
        f = open(dirpath + path + '.pickle', 'wb')
        pickle.dump(self.hist,f)
        f.close()
        self.Net.save_weights(dirpath + path + '.h5')

    def load(self, path):
        f = open(dirpath + path + '.pickle', 'rb')
        self.hist = pickle.load(f)
        f.close()
        self.Net.load_weights(dirpath + path + '.h5')

        self.curr_epoch = self.hist['iterations'][0][-1]
