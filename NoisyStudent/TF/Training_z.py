from Networks_z import *

import torch
from torchvision import transforms

from sklearn.model_selection import train_test_split

from dynamicDataLoader import Dynamic_dataloader_subaru_params

dataset_loader = torch.utils.data.DataLoader(Dynamic_dataloader_subaru_params(\
    transforms.Normalize(mean=(0.0598,0.0123,0.0207,0.0311,0.0403),\
    std=(1.01,0.0909,0.193,0.357,0.552)),cut_size=16), \
    batch_size=10000, shuffle=True, num_workers=1, pin_memory=True)

img,z,sigma = next(iter(dataset_loader))
img = np.transpose(img.numpy(),(0,2,3,1))
params = np.vstack([z.numpy(), sigma.numpy()]).T

x_train, x_test, y_train, y_test = train_test_split(img,params,test_size=0.2)

def TrainTeacher(epochs,teach_path = './records/test', load = False):
   
    Teacher_Net = Network_z(np.shape(x_train[0]), 2, noise=False)
    if load:
        Teacher_Net.load(path=teach_path)
    Teacher_Net.train(x_train, y_train, x_test, y_test, epochs)
    Teacher_Net.save(path=teach_path)


def TrainStudent(trainsplit, epochs, teach_path, student_path, load = False):
    Teacher_Net = Network(Data.input_shape, Data.num_classes, noise=False)
    Teacher_Net.load(path=teach_path)

    Student_Net = Network(Data.input_shape, Data.num_classes, noise=True)
    if load:
        Student_Net.load(path=student_path)
    Student_Net.train(x_train_new, y_train_new, x_test, y_test, epochs)
    Student_Net.save(path=student_path)


