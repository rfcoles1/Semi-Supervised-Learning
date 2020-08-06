import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pickle
from collections import defaultdict

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as layers
from tensorflow.keras import backend as K
import tensorflow_probability as tfp

import sklearn

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

def get_negative_mask(batch_size):
    negative_mask = np.ones((batch_size, 2*batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i,i] = 0
        negative_mask[i, i+batch_size] = 0
    #return tf.constant(negative_mask)
    return negative_mask

def cos_sim_dim1(x,y):
    cos_sim = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
    return cos_sim(x,y)

def cos_sim_dim2(x,y):
    cos_sim = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)
    return cos_sim(tf.expand_dims(x,1), tf.expand_dims(y,0))

class CustomAugment(object):
    def __call__(self, sample):        
        # Random flips
        sample = self._random_apply(tf.image.flip_left_right, sample, p=0.5)
        
        # Randomly apply transformation (color distortions) with probability p.
        sample = self._random_apply(self._color_jitter, sample, p=0.8)
        #sample = self._random_apply(self._color_drop, sample, p=0.2)

        return sample

    def _color_jitter(self, x, s=1):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8*s)
        x = tf.image.random_contrast(x, lower=1-0.8*s, upper=1+0.8*s)
        #x = tf.image.random_saturation(x, lower=1-0.8*s, upper=1+0.8*s)
        #x = tf.image.random_hue(x, max_delta=0.2*s)
        x = tf.clip_by_value(x, 0, 1)
        return x
    
    def _color_drop(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 1, 5])
        return x
    
    def _random_apply(self, func, x, p):
        return tf.cond(
          tf.less(tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32),
                  tf.cast(p, tf.float32)),
          lambda: func(x),
          lambda: x)


class Network():
    def __init__(self, input_shape, noise=False):
        
        self.dirpath = 'records_z/'
        if not os.path.exists(self.dirpath):
            os.makedirs(self.dirpath)
        
        self.lr = 0.001
        self.temp = 0.1
        self.batch_size = 64
        self.z_size = 64
        self.negative_mask = get_negative_mask(self.batch_size)
        self.input_shape = input_shape

        self.hist = defaultdict(list)
        self.chkpt = 1
        self.res_curr_epoch = 0
        self.mlp_curr_epoch = 0

        self.data_aug = keras.Sequential([layers.Lambda(CustomAugment())])

        self.inp1 = layers.Input(input_shape)      
        self.inp2 = layers.Input(self.z_size)
        self.ResNet = tf.keras.models.Model(self.inp1, self.resnet(self.inp1))
        self.Mlp = tf.keras.models.Model(self.inp2, self.mlp(self.inp2))
        self.SimClr = tf.keras.models.Model(self.inp1, self.mlp(self.resnet(self.inp1)))

        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(\
            from_logits=True, reduction=tf.losses.Reduction.SUM)
        self.optimizer = tf.keras.optimizers.SGD()

        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss",\
            patience=3, verbose=3, restore_best_weights=True)
        
        self.Mlp.compile(loss=tf.keras.losses.MSE,
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr))

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

    def mlp(self, x, hid1=64):
        y = layers.Dense(hid1, activation=layers.LeakyReLU(alpha=0.1))(x)
        y = layers.Dense(1)(y)
        return y
    


    def train_res(self, x_train, epochs, verbose=2):
        it = 1 
        num_batches = int(np.floor(len(x_train)/self.batch_size)) 

        batch_losses = np.zeros(num_batches)
        epoch_losses = np.zeros(self.chkpt)

        for it in range(epochs+1):
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
                self.res_curr_epoch += self.chkpt
                
                print('Epoch %d - Time Taken %f' % (self.res_curr_epoch, time.time()-start))
                print('Train Loss %f' % np.mean(batch_losses))

                self.hist['res_epochs'].append(self.res_curr_epoch)
                self.hist['res_train_loss'].append(epoch_losses)

    def _train_step(self, x_train_i, x_train_j):
        with tf.GradientTape() as tape:

            zi = self.ResNet(x_train_i)
            zj = self.ResNet(x_train_j)

            zi = tf.math.l2_normalize(zi, axis=1)
            zj = tf.math.l2_normalize(zj, axis=1)
            
            l_pos = cos_sim_dim1(zi, zj)
            l_pos = tf.reshape(l_pos, (self.batch_size,1))
            l_pos /= self.temp
            
            
            negatives = tf.concat([zj, zi], axis=0)
            
            loss = 0
            for positives in [zi, zj]:
                
                labels = tf.zeros(self.batch_size, dtype=tf.int32)
                l_neg = cos_sim_dim2(positives, negatives)
                l_neg = tf.boolean_mask(l_neg, self.negative_mask)
                l_neg = tf.reshape(l_neg, (self.batch_size, -1))
                l_neg /= self.temp
                
                logits = tf.concat([l_pos, l_neg], axis=1)
                loss += self.criterion(y_pred=logits, y_true=labels)
                    
            loss = loss/(2*self.batch_size)

            gradients = tape.gradient(loss, self.ResNet.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.ResNet.trainable_variables))
            
            return loss

    def train_mlp(self, x_train, y_train, x_test, y_test, epochs):
        x_train_rep = self.ResNet(x_train)
        x_test_rep = self.ResNet(x_test)
        
        history = self.Mlp.fit(x_train_rep,y_train,
            batch_size = self.batch_size,
            epochs = epochs,
            verbose = 2,
            validation_data = (x_test_rep, y_test))

        #self.hist['mlp_epochs'].append(
        self.hist['mlp_train_loss'].append(history.history['loss'])
        self.hist['mlp.test_loss'].append(history.history['val_loss'])

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
    


