import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage import transform
from scipy.ndimage.filters import gaussian_filter

seed = 0
np.random.seed(seed)

class Augmenter():
    def __init__(self, number=1):
       
        self.number = number
        self.transforms = ['rotate', 'translateX', 'translateY', 'shear',\
                            'noise', 'filter', 'lineX', 'lineY']
        
        self.func = {
            "rotate": lambda x, param: transform.rotate(x, param*360),
            "translateX": lambda x, param: transform.warp(x,\
                transform.AffineTransform(translation=(param*1 - 1,0))),
            "translateY": lambda x, param: transform.warp(x,\
                transform.AffineTransform(translation=(0, param*1 - 1))),
            "shear": lambda x, param: transform.warp(x,\
                transform.AffineTransform(shear=(param -0.5))),
            "noise": lambda x, param: np.clip(x + np.random.normal(0,param*0.1,x.shape),0,1),
            "filter": lambda x, param: gaussian_filter(x, 1),
            "lineX": lambda x, param: self.removeX(x,param),
            "lineY": lambda x, param: self.removeY(x,param)
        }

    def removeX(self,x,param):
        x[:, int(param*27): int(param*27 + 1)] = 0
        return x

    def removeY(self,x,param):
        x[int(param*27): int(param*27 + 1), :] = 0
        return x

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
            mag = m#self.ranges[op][m]
            x = operation(x, mag)
        return x

    def get_transforms(self):
        M = np.random.rand(self.number)
        ops = np.random.choice(self.transforms,self.number)
        return [(op, m) for (op, m) in zip(ops,M)]

    def single_transform(self, x, op, M = -1):
        if M == -1:
            M = np.random.rand()
        operation = self.func[op]
        x = operation(x, M)
        return x
