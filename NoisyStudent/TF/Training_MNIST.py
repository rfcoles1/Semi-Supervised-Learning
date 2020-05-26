from Networks import *
from Plotting import *

import sys
sys.path.insert(0, '../')
from Data import *

test_frac = 0.2
Data = Loader(test_frac) #assume 20% of data will be kept for testing
x_test, y_test = Data.get_test()

def TrainTeacher(trainsplit, epochs, teach_path, load = False):
    Data.reset()
    x_train, y_train = Data.get_train(trainsplit)

    Teacher_Net = Network(Data.input_shape, Data.num_classes, noise=False)
    if load:
        Teacher_Net.load(path=teach_path)
    Teacher_Net.train(x_train, y_train, x_test, y_test, epochs)
    Teacher_Net.save(path=teach_path)


def TrainStudent(trainsplit, epochs, teach_path, student_path, load = False):
    Teacher_Net = Network(Data.input_shape, Data.num_classes, noise=False)
    Teacher_Net.load(path=teach_path)

    Data.reset()
    x_train, y_train = Data.get_train(trainsplit)
    x_pred, _ = Data.get_train(0.99 - test_frac - trainsplit) 
    y_pred = Teacher_Net.predict(x_pred)

    x_train_new = np.concatenate([x_train,x_pred])
    y_train_new = np.concatenate([y_train,y_pred])

    Student_Net = Network(Data.input_shape, Data.num_classes, noise=True)
    if load:
        Student_Net.load(path=student_path)
    Student_Net.train(x_train_new, y_train_new, x_test, y_test, epochs)
    Student_Net.save(path=student_path)


def TrainStudent_Aug(trainsplit, epochs, teach_path, student_path, num_ops=2, load = False):
    Aug = Augmenter(num_ops)

    Teacher_Net = Network(Data.input_shape, Data.num_classes, noise=False)
    Teacher_Net.load(path=teach_path)

    Data.reset()
    x_train, y_train = Data.get_train(trainsplit)
    x_pred, _ = Data.get_train(0.99 - test_frac - trainsplit) 
    y_pred = Teacher_Net.predict(x_pred)

    x_train_aug = np.zeros_like(x_train)
    for i in range(len(x_train)):
        tmp = np.reshape(x_train[i],[28,28])
        tmp = Aug.transform(tmp)
        tmp = np.reshape(tmp, [28,28,1])
        x_train_aug[i] = tmp

    x_pred_aug = np.zeros_like(x_pred)
    for i in range(len(x_pred)):
        tmp = np.reshape(x_pred[i],[28,28])
        tmp = Aug.transform(tmp)
        tmp = np.reshape(tmp, [28,28,1])
        x_pred_aug[i] = tmp

    x_train_new = np.concatenate([x_traini_aug,x_pred_aug])
    y_train_new = np.concatenate([y_train,y_pred])

    Student_Net = Network(Data.input_shape, Data.num_classes, noise=True)
    if load:
        Student_Net.load(path=student_path)
    Student_Net.train(x_train_new, y_train_new, x_test, y_test, epochs)
    Student_Net.save(path=student_path)


