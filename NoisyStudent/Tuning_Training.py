import kerastuner as kt

from Tuning_Networks import *

epochs = int(sys.argv[1])
trials = int(sys.argv[2])
datafrac = float(sys.argv[3])

tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=trials,
    directory='Tuning_Results')

Data = Loader(0.2, 'load_z')
x_train, y_train = Data.get_train(datafrac)
x_test, y_test = Data.get_test()

x_train = cutimgs(x_train)
x_test = cutimgs(x_test)

tuner.search(x_train, y_train, epochs=epochs,\
    validation_data=(x_test,y_test))

best_params = tuner.get_best_hyperparameters(1)[0]

print('Batch_size: ' + str(best_params['batch_size']))
print('Learning_Rate: ' + str(best_params['learning_rate']))
print('Alpha: ' + str(best_params['batch_size']))
print('Layer 1 neurons: ' + str(best_params['neurons1']))
print('Layer 2 neurons: ' + str(best_params['neurons2']))
print('Layer 3 neurons: ' + str(best_params['neurons3']))
print('Dropout_Rate: ' + str(best_params['dropout_rate']))





