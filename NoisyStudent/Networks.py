import warnings
warnings.filterwarnings('ignore')

import os, sys
import numpy as np
import pickle
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as layers
from tensorflow.keras import backend as K

import wandb
from wandb.keras import WandbCallback

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {'loss':[], 'val_loss':[]}

    def on_batch_end(self, batch, logs={}):
        self.history['loss'].append(logs.get('loss'))

class Network():
    def __init__(self):

        self.hist = defaultdict(list)
        self.curr_epoch = 0

    def predict(self, x_test):
        preds = self.Net.predict(x_test)
        return preds
   
    def evaluate(self, x_test, y_test):
        return self.Net.evaluate(x_test,y_test)

    def history(self):
        return self.hist

    def save(self, path):
        f = open(self.dirpath + path + '.pickle', 'wb')
        pickle.dump(self.hist,f)
        f.close()
        self.Net.save_weights(self.dirpath + path + '.h5')

    def load(self, path):
        f = open(self.dirpath + path + '.pickle', 'rb')
        self.hist = pickle.load(f)
        f.close()
        self.Net.load_weights(self.dirpath + path + '.h5')
        self.curr_epoch = self.hist['epochs'][-1][-1]

class AutoEncoder():
    def __init__(self):

        self.hist = defaultdict(list)
        self.curr_epoch = 0

    def predict(self, x_test):
        preds = self.Net.predict(x_test)
        return preds
   
    def evaluate(self, x_test, y_test):
        return self.Net.evaluate(x_test,y_test)

    def history(self):
        return self.hist

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
        self.curr_epoch = self.hist['epochs'][-1][-1]
