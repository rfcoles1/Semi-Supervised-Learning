import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage import transform
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import rotate

seed = 0
np.random.seed(seed)

class Augmenter():
    def __init__(self, imgsize = 32, N=1, transforms='All', M=[[0,1]]):
        
        self.imsize = imgsize
        self.N = N #number of transformations

        self.func = {
            "rotate": lambda x, mag: rotate(x, mag, axes=(0,1), reshape=False),
            "translateX": lambda x, mag: transform.warp(x,\
                transform.AffineTransform(translation=(mag, 0))),
            "translateY": lambda x, mag: transform.warp(x,\
                transform.AffineTransform(translation=(0, mag))),
            "shear": lambda x, mag: transform.warp(x,\
                transform.AffineTransform(shear=(mag))),
            "noise": lambda x, mag: np.clip(x + np.random.normal(0,mag,x.shape),0,1),
            "onenoise": lambda x, mag: self.oneChannelNoise(x, mag),
            "filter": lambda x, mag: gaussian_filter(x, mag),
            "flatscale": lambda x, mag: self.scaleChannelMaxFlat(x, mag),
            "noisescale": lambda x, mag: self.scaleChannelMaxNoise(x, mag)}
            
        self.ranges = {
                "rotate": [0,360],
                "translateX": [0, self.imsize/2],
                "translateY": [0, self.imsize/2],
                "shear": [0,1],
                "noise": [0,1],
                "onenoise": [0,1],
                "filter": [0,1],
                "flatscale": [0, 0.05],
                "noisescale": [0, 0.05]}

        if transforms == 'All':#default then use all functions
            self.transforms = ['rotate', 'translateX', 'translateY', 'shear',\
                            'noise', 'onenoise', 'filter', 'flatscale', 'noisescale']
        else: #specifies some functions
            self.transforms = transforms
            if len(transforms)==len(M):#if the correct number of ranges are also specificed
                self.ranges = {}
                for i in range(len(transforms)):
                    print(transforms[i], M[i]) 
                    self.ranges[transforms[i]] = M[i]
      
    def oneChannelNoise(self, x, mag):
        noise = np.random.normal(0,mag,x.shape)
        dims = np.shape(x)[-1]
        
        oh_channel = np.zeros(dims)
        oh_channel[np.random.randint(0,dims)]=1

        for i in range(dims):
            noise[:,:,i] *= oh_channel[i]

        return np.clip(x + noise, 0,1)

    def scaleChannelMaxFlat(self, x, mag):
        noise = np.ones_like(x)
        dims = np.shape(x)[-1]

        for i in range(dims):
            thismax = np.max(x[:,:,i])
            noise[:,:,i] *= thismax*mag

        return np.clip(x + noise, 0,1)

    def scaleChannelMaxNoise(self, x, mag):
        noise = np.zeros_like(x)
        dims = np.shape(x)[-1]
        shape = np.shape(x[:,:,0])

        for i in range(dims):
            thismax = np.max(x[:,:,i])
            noise[:,:,i] = np.random.normal(0,mag*thismax, shape)

        return np.clip(x + noise, 0,1)

    #def transform_set(self,x):
    def __call__(self,x):
        x_aug = np.zeros_like(x)
        for i in range(len(x)):
            x_aug[i] = self.transform(x[i])
        return x_aug

    def transform(self, x):
        operations = self.get_transforms()
        for (op, m) in operations:
            operation = self.func[op]
            op_min, op_max = self.ranges[op]
            op_m = m*(op_max-op_min) + op_min
            x = operation(x, op_m)
        return x

    def get_transforms(self):
        ops = np.random.choice(self.transforms,self.N)     
        M = np.random.rand(self.N)
        return [(op, m) for (op, m) in zip(ops,M)]

    def single_transform(self, x, op, M = -1):
        if M == -1:
            M = np.random.rand()
        operation = self.func[op]
        x = operation(x, M)
        return x


def removeX(x,param):
    x[:, int(param*(self.imsize)): int(param*self.imsize + 1)] = 0
    return x

def removeY(x,param):
    x[int(param*self.imsize): int(param*self.imsize + 1), :] = 0
    return x


