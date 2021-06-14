import warnings 
warnings.filterwarnings('ignore')

import sys
import numpy as np
import pickle
import copy

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage import transform
from scipy.ndimage.filters import gaussian_filter

from astropy.cosmology import FlatLambdaCDM, w0waCDM
import astropy.units as u

seed = 0
np.random.seed(seed)

def trim_channels(data,channels):
    #channels in the format [1,2,3]
    return data[:,:,:,channels]

def zero_channels(data,channels):
    #channels in the format [1,2,3]
    data_tmp = copy.deepcopy(data)
    data_tmp[:,:,:,channels] = 0
    return data_tmp

def class_encoder(data, n_bins, y_min=0, y_max=1):
    #assume labels are normalize to be between 0 and 1
    bins = np.linspace(y_min,y_max,n_bins,endpoint=False)
    digitized = np.digitize(data,bins)
    return digitized, bins

def get_new_labels(x_test, pred, threshold=0):
    maxind = np.argmax(pred, axis=1)
    if threshold == 0:
        return x_test, maxind
    else:
        maxval = np.max(pred, axis=1)
        idx = np.where(maxval>threshold)
        return x_test[idx], maxind[idx]



def load_test():
    try:
        img, z, _, _, _ = pickle.load(open("../Data/data_2500.pickle","rb"))
        out_size = 1
        return img, z, np.shape(img[0]), out_size 
    except:
        print("Could not load galaxy data")

def load_full():
    try:
        img, z, _, _, _ = pickle.load(open("../Data/data_full.pickle","rb"))
        out_size = 1
        return img, z, np.shape(img[0]), out_size 
    except:
        print("Could not load galaxy data")

def load_dist():
    try:
        img, z, z_sig, dist, dist_sig = pickle.load(open("../Data/data_dist.pickle","rb"))
        out_size = 1
        return img, dist, np.shape(img[0]), out_size 
    except:
        print("Could not load galaxy data")

def rebalance_dataset(img, z, z_min, z_max, bins=20):
    bin_amount = int(len(z)/bins)
    idx = [i for i, z in enumerate(z) if z > z_max]
    img = np.delete(img, idx,axis=0)
    z = np.delete(z, idx)
   
    idx = [i for i, z in enumerate(z) if z < z_min]
    img = np.delete(img, idx,axis=0)
    z = np.delete(z, idx)
   
    n, bin_edges = np.histogram(z,bins)
    idxs = np.digitize(z, bin_edges)-1
    new_imgs = [] 
    new_z = []
    for j in range(len(n)):
        idx = [ix for ix, i in enumerate(idxs) if i==j]
        if n[j] > bin_amount:
            idx = np.random.choice(idx, bin_amount, replace=False)
            new_imgs.append(img[idx,:,:,:])
            new_z.append(z[idx])
        else:
            idx = np.hstack([idx, np.random.choice(idx,bin_amount-n[j])])
            new_imgs.append(img[idx,:,:,:])
            new_z.append(z[idx])
                    
    img = [item for sublist in new_imgs for item in sublist]
    img = np.array(img)
    z = [item for sublist in new_z for item in sublist]
    z = np.array(z)

    return img, z    


def get_dist_flat(z, err_per=0):
    model = FlatLambdaCDM(H0=73*u.km/u.s/u.Mpc, Om0 = 0.3)
    dist = model.angular_diameter_distance(z).value
    error = (err_per*dist)*np.random.normal(0,1,np.shape(z))
    return dist+error

def get_dist_w0(z, err_per=0):
    model = w0waCDM(H0=68*u.km/u.s/u.Mpc, Om0=0.28, Ode0=0.73, w0=-0.9, wa=0.2)
    dist = model.angular_diameter_distance(z).value
    error = (err_per*dist)*np.random.normal(0,1,np.shape(z))
    return dist+error


class Loader():
    def __init__(self, test_per, dat, balanced=False, dist=None,err=0.07):
        self.datasets = {
            "load_test": load_test(),
            "load_full": load_full(),
            "load_dist": load_dist()}

        x, y, self.shape, self.num_out = self.datasets[dat]
        
        self.dims = self.shape[-1]
        self.scaler = Scaler(self.dims)
        #x_scaled = np.arcsinh(x)
        #x_scaled = self.scaler.minmax_x(x_scaled)

        if dist=='Flat':
            self.redshifts = y
            y = get_dist_flat(y,err)
        if dist=='w0':
            self.redshifts = y
            y = get_dist_w0(y,err)

        y_scaled = self.scaler.minmax_y(y)

        self.test_per = test_per
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x,y_scaled, test_size=self.test_per, random_state=seed)
        
        if balanced==True:
            bins = 10
            z_min = 0.5
            z_max = 6.5
            z_min = self.scaler.norm(z_min)
            z_max = self.scaler.norm(z_max)
            self.x_train, self.y_train = rebalance_dataset(\
                                        self.x_train, self.y_train, z_min, z_max, bins)
       
        self.reset()

    def reset(self):
        self.percentage_returned = self.test_per
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
            train_test_split(self.x_stored, self.y_stored,\
                test_size=(1-scaled_train_per), random_state=seed)
        
        self.percentage_returned += train_per
        print('%.2f%% of the data has been used' %(100*self.percentage_returned))
        return x_train, y_train

    def get_full_train(self):
        return self.x_train, self.y_train

    def get_test(self):
        return self.x_test, self.y_test
            

class Scaler():
    def __init__(self, dims):
        self.dims = dims
        self.x_min = np.zeros(self.dims)
        self.x_max = np.zeros(self.dims)
        self.y_min = 0
        self.y_max = 0

    def minmax_x(self, x, bychannels=True):
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

    def minmax_y(self,y):
        self.y_min = np.min(y)
        self.y_max = np.max(y)
        y = (y-self.y_min)/(self.y_max-self.y_min)
        return y
    
    def norm(self, y):
        return (y-self.y_min)/(self.y_max-self.y_min)
