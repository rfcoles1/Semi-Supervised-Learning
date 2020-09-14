import warnings
warnings.filterwarnings('ignore')

import os, sys
import time
import numpy as np
import itertools
import pickle
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as layers
from tensorflow.keras import backend as K
import tensorflow_probability as tfp

import sklearn

sys.path.insert(1, '../Utils')
from metrics import *
from datasets import *
from augment import *


def get_negative_mask(batch_size):
    negative_mask = np.ones((batch_size, 2*batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i,i] = 0
        negative_mask[i, i+batch_size] = 0
    return tf.constant(negative_mask)


def dot_sim_dim1(x,y):
    return tf.matmul(tf.expand_dims(x,1), tf.expand_dims(y,2))

def dot_sim_dim2(x,y):
    return tf.tensordot(tf.expand_dims(x,1), tf.expand_dims(tf.transpose(y), 0), axes=2)

def cos_sim_dim1(x,y):
    cos_sim = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
    return cos_sim(x,y)

def cos_sim_dim2(x,y):
    cos_sim = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)
    return cos_sim(tf.expand_dims(x,1), tf.expand_dims(y,0))


class Network():
    def __init__(self, input_shape, noise=False):
        
        self.dirpath = 'records_z/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        
        self.temp = 0.1
        self.batch_size = 64
        self.z_size = 64
        self.negative_mask = get_negative_mask(self.batch_size)
        self.input_shape = input_shape

        self.hist = defaultdict(list)
        self.chkpt = 1
        self.simclr_curr_epoch = 0
        self.mlp_curr_epoch = 0

        #self.data_aug = keras.Sequential([layers.Lambda(CustomAugment())])
        self.data_aug = Augmenter(2)
    
        self.inp1 = layers.Input(input_shape)      
        self.inp2 = layers.Input(self.z_size)
        
        self.SimClr = tf.keras.models.Model(self.inp1, self.mlp(self.tf_resnet(self.inp1)))
        self.Mlp = tf.keras.models.Model(self.inp2, self.mlp(self.inp2))
       
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(\
            from_logits=True, reduction=tf.losses.Reduction.SUM)
        
        lr_decayed_fn = tf.keras.experimental.CosineDecay(\
            initial_learning_rate=0.1, decay_steps=1000)
        self.optimizer = tf.keras.optimizers.Adam(lr_decayed_fn)

        self.es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",\
            patience=3, verbose=3, restore_best_weights=True)
        
        self.Mlp.compile(loss=tf.keras.losses.MSE,
            optimizer=self.optimizer)


    def tf_resnet(self,x):
        base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,\
            input_shape=self.input_shape)
        base_model.trainabe = True
        
        x = base_model(x, training=True)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dense(128, activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dense(self.z_size)(x)
        return x
    
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
        x = layers.Dense(128, activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dense(self.z_size)(x)
        return x

    def mlp(self, x):
        x = layers.Dense(256, activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dense(128, activation=layers.LeakyReLU(alpha=0.1))(x)
        x = layers.Dense(1)(x)
        return x
    


    def train_augment(self, x_train, epochs):
        num_batches = int(np.floor(len(x_train)/self.batch_size)) 

        batch_losses = np.zeros(num_batches)
        epoch_losses = np.zeros(self.chkpt)

        for it in range(epochs):
            start = time.time()
            x_train = sklearn.utils.shuffle(x_train)
            for i in range(num_batches):
                batch_x = x_train[i*self.batch_size:(i+1)*self.batch_size]

                x_train_i = self.data_aug(batch_x)
                x_train_j = self.data_aug(batch_x)
                
                loss = self._train_step(x_train_i, x_train_j)
                 
                batch_losses[i] = loss
            
            epoch_losses[it%self.chkpt] = np.mean(batch_losses) 
            
            if it % self.chkpt == 0:
                self.simclr_curr_epoch += self.chkpt
                
                print('Epoch %d - Time Taken %f' % (self.simclr_curr_epoch, time.time()-start))
                print('Train Loss %f' % np.mean(batch_losses))

                self.hist['sim_epochs'].append(self.simclr_curr_epoch)
                self.hist['sim_loss'].append(epoch_losses)

            it += 1

    def train_super(self, x_train, y_train, epochs):
        num_batches = int(np.floor(len(x_train)/self.batch_size)) 

        batch_losses = np.zeros(num_batches)
        epoch_losses = np.zeros(self.chkpt)

        for it in range(epochs):
            start = time.time()
            x_train,y_train = sklearn.utils.shuffle([x_train,y_train])
            for i in range(num_batches):
                batch_x = x_train[i*self.batch_size:(i+1)*self.batch_size]
                batch_y = y_train[i*self.batch_size:(i+1)*self.batch_size]

                ### METHOD TO GET PAIRS OF SIMILAR DATA
                ### Will need to return two lists for each point
                pairs = get_pairs(batch_y)
                left,right = np.hsplit(pairs)
                x_train_i = batch_x[left]
                x_train_j = batch_x[right]

                loss = self._train_step(x_train_i, x_train_j)
                 
                batch_losses[i] = loss
            
            epoch_losses[it%self.chkpt] = np.mean(batch_losses) 

            if it % self.chkpt == 0:
                self.simclr_curr_epoch += self.chkpt
                
                print('Epoch %d - Time Taken %f' % (self.simclr_curr_epoch, time.time()-start))
                print('Train Loss %f' % np.mean(batch_losses))

                self.hist['sim_epochs'].append(self.simclr_curr_epoch)
                self.hist['sim_loss'].append(epoch_losses)

            it += 1

    def _train_step(self, x_train_i, x_train_j):
        with tf.GradientTape() as tape:
            zi = self.SimClr(x_train_i)
            zj = self.SimClr(x_train_j)


            zi = tf.math.l2_normalize(zi, axis=1)
            zj = tf.math.l2_normalize(zj, axis=1)
            
            l_pos = dot_sim_dim1(zi, zj)
            l_pos = tf.reshape(l_pos, (self.batch_size,1))
            l_pos /= self.temp
            negatives = tf.concat([zj, zi], axis=0)
            
            loss = 0
            for positives in [zi, zj]:
                
                labels = tf.zeros(self.batch_size, dtype=tf.int32)
                l_neg = dot_sim_dim2(positives, negatives)
                l_neg = tf.boolean_mask(l_neg, self.negative_mask)
                l_neg = tf.reshape(l_neg, (self.batch_size, -1))
                l_neg /= self.temp
                logits = tf.concat([l_pos, l_neg], axis=1)
                
                loss += self.criterion(y_pred=logits, y_true=labels)
            loss = loss/(2*self.batch_size)

            gradients = tape.gradient(loss, self.SimClr.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.SimClr.trainable_variables))
            
            return loss

    def train_mlp(self, x_train, y_train, x_test, y_test, epochs):
        x_train_rep = self.ResNet(x_train)
        x_test_rep = self.ResNet(x_test)
        
        History = self.Mlp.fit(x_train_rep,y_train,
            batch_size = self.batch_size,
            epochs = epochs,
            verbose = 2,
            validation_data = (x_test_rep, y_test))

        epochs_arr = np.arange(self.mlp_curr_epoch, self.mlp_curr_epoch_epochs, 1)
        iterations = np.ceil(np.shape(x_train)[0]/self.batch_size)

        self.hist['epochs'].append(epochs_arr)
        self.hist['iterations'].append(epochs_arr*iterations)
        self.hist['train_MSE'].append(History.history['loss'])
        self.hist['test_MSE'].append(History.history['val_loss'])

        self.mlp_curr_epoch += epochs

    def predict(self, x_test):
        x_test_rep = self.ResNet(x_test)
        
        preds = self.Mlp.predict(x_test_rep,
                        batch_size=self.batch_size,
                        verbose=0)
        return preds

    def evaluate(self, x_test, y_test):
        x_test_rep = self.ResNet(x_test)
 
        return self.Mlp.evaluate(x_test_rep,y_test)

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
    
#returns indices of points that ~should~ be similar redshift values
#computed such that the set of pairs with the smallest summed distance is returned
#really needs to be optimized lol 
def get_pairs(elements):
    num = len(elements)
    num_pairs = int(np.ceil(num)/2)
    indices = np.arange(0,num,1)
    
    pairs = list(itertools.combinations(indices,2))
    pairs = [list(p) for p in pairs]
    
    combs = list(itertools.combinations(pairs,num_pairs))
    combs = [list(p) for p in combs]
    print('here') 
    flat_combs = np.array(combs).reshape(-1,num)
    
    sets = [True if len(np.unique(p))==num else False for p in flat_combs]
    idxs = [i for i,b in enumerate(sets) if b]
    print('now here')
    pairsets = np.array(combs)[idxs]
    
    indices = np.digitize(pairsets.ravel(),indices, right=True)
    values = elements[indices].reshape(pairsets.shape)
    print('one last time')
    min_set_idx = np.argmin(np.sum(abs(np.diff(values,axis=2)),axis=1))
    return pairsets[min_set_idx], values[min_set_idx]
