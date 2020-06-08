import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skimage import transform
from scipy.ndimage.filters import gaussian_filter

seed = 0
np.random.seed(seed)

def get_mnist():
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras import backend as K
    
    img_rows, img_cols = 28, 28
    num_classes = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test), axis=0 )

    if K.image_data_format() == 'channels_first':
        x = x.reshape(x.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x = x.reshape(x.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x = x.astype('float32')
    x /= 255

    y = tf.keras.utils.to_categorical(y, num_classes)

    return x, y, input_shape, num_classes

def get_z_dataset(N = 1000, sigma=False):
    from dynamicDataLoader import Dynamic_dataloader_subaru_params

    dataset_loader = torch.utils.data.DataLoader(Dynamic_dataloader_subaru_params(\
        transforms.Normalize(mean=(0.0598,0.0123,0.0207,0.0311,0.0403),\
        std=(1.01,0.0909,0.193,0.357,0.552)),cut_size=16), \
        batch_size=N, shuffle=True, num_workers=1, pin_memory=True)

    img,z,sigma = next(iter(dataset_loader))
    img = np.transpose(img.numpy(),(0,3,1,2))
    
    if sigma:
        params = np.hstack([z.numpy(), sigma.numpy()])
        out_size = 2
    else:
        params = z.numpy()
        out_size = 1

    return img, params, np.shape(img[0]), out_size

def load_z_dataset():
    import pickle
    img, params = pickle.load(open("data.pickle","rb"))
    out_size = 1
    return img, params, np.shape(img[0]), out_size 

class Loader():
    def __init__(self, test_per, dat):
        
        self.datasets = {
            "MNIST": get_mnist(),
            #"get_z": get_z_dataset(),
            "load_z": load_z_dataset()}

        x, y, self.input_shape, self.num_classes = self.datasets[dat]

        self.test_per = test_per
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(x,y, test_size=self.test_per, random_state=seed)
                    
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
               
class Augmenter():
    def __init__(self, number=1):
       
        self.number = number
        self.transforms = ['rotate', 'translateX', 'translateY', 'shear',\
                            'noise', 'filter', 'lineX', 'lineY']
        
        self.func = {
            "rotate": lambda x, param: transform.rotate(x, param*360),
            "translateX": lambda x, param: transform.warp(x,\
                transform.AffineTransform(translation=(param*10 - 5,0))),
            "translateY": lambda x, param: transform.warp(x,\
                transform.AffineTransform(translation=(0, param*10 - 5))),
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
