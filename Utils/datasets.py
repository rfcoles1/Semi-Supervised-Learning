import warnings 
warnings.filterwarnings('ignore')

import sys
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage import transform
from scipy.ndimage.filters import gaussian_filter

seed = 0
np.random.seed(seed)

def load_z_dataset():
    try:
        img, z, _, _, _ = pickle.load(open("../Data/data_coord.pickle","rb"))
        #img, z = pickle.load(open("../Data/databig.pickle","rb"))

        out_size = 1
        
        idx = [i for i, z in enumerate(z) if z > 3.5]
        img = np.delete(img, idx,axis=0)
        z = np.delete(z, idx)
        
        idx = [i for i, z in enumerate(z) if z < 0.5]
        img = np.delete(img, idx,axis=0)
        z = np.delete(z, idx)

        n, bin_edges = np.histogram(z,20)
        idxs = np.digitize(z, bin_edges)-1
        new_imgs = [] 
        new_z = []
        for j in range(len(n)):
            idx = [ix for ix, i in enumerate(idxs) if i==j]
            if n[j] > 500:
                idx = np.random.choice(idx, 500, replace=False)
                new_imgs.append(img[idx,:,:,:])
                new_z.append(z[idx])
            else:
                idx = np.hstack([idx, np.random.choice(idx,500-n[j])])
                new_imgs.append(img[idx,:,:,:])
                new_z.append(z[idx])
                 
        img = [item for sublist in new_imgs for item in sublist]
        img = np.array(img)
        z = [item for sublist in new_z for item in sublist]
        z = np.array(z)
        
        return img, z, np.shape(img[0]), out_size 
    
    except:
        print("Could not load galaxy data")

class Scaler():
    def __init__(self, dims):
        self.dims = dims
        self.x_min = np.zeros(self.dims)
        self.x_max = np.zeros(self.dims)
        self.y_min = 0
        self.y_max = 0

    def minmax_img(self, x, bychannels=True):
        if bychannels:
            for i in range(self.dims):
                self.x_min[i] = np.min(x[:,:,:,i])
                self.x_max[i] = np.max(x[:,:,:,i])
                x[:,:,:,i] = (x[:,:,:,i] - self.x_min[i])/(self.x_max[i] - self.x_min[i])
        else:
            self.x_min[0] = np.min(x)
            self.x_max[0] = np.max(x)
            x = (x-self.x_min[0])/(self.x_max[0]-self.x_min[0])
        return x

    def arcsinh(self, x):
        return np.arcsinh(x)

    def minmax_z(self,y):
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        y = (y-self.y_min)/(self.y_max-self.y_min)
        return y

class Loader():
    def __init__(self, test_per, dat):
        
        self.datasets = {
            "load_z": load_z_dataset()}

        x, y, self.shape, self.num_out = self.datasets[dat]

        self.dims = self.shape[-1]
        self.scaler = Scaler(self.dims)
        #x_scaled = self.scaler.arcsinh(x)
        #x_scaled = self.scaler.minmax_img(x_scaled)
        y_scaled = self.scaler.minmax_z(y)

        self.test_per = test_per
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x,y_scaled, test_size=self.test_per, random_state=seed)
                    
        self.reset()

    def reset(self):
        self.percentage_returned = self.test_per
        self.x_returned = np.array([])#np.empty_like(self.x_train)
        self.y_returned = np.array([])#np.empty_like(self.y_train)
        self.x_stored = np.copy(self.x_train)
        self.y_stored = np.copy(self.y_train)

    def get_train(self, train_per):
        if self.percentage_returned + train_per > 1:
            train_per = 1 - self.percentage_returned
            print('Only have train_per %.2f%% data left available' %(100*train_per))
      
        if train_per < 0.0:
            print('No data remaining')
            return 

        scaled_train_per = train_per/(1.0 - self.percentage_returned)

        x_train, self.x_stored, y_train, self.y_stored = \
            train_test_split(self.x_stored, self.y_stored, test_size=(1-scaled_train_per), random_state=seed)
        
        self.percentage_returned += train_per
        print('%.2f%% of the data has been used' %(100*self.percentage_returned))

        return x_train, y_train

    def get_full_train(self):
        return self.x_train, self.y_train

    def get_test(self):
        return self.x_test, self.y_test
               
