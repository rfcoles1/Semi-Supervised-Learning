from Networks import *

sys.path.insert(1, '../Utils')
from datasets import *
from augment import *
        
def build_model(hp):
    input_shape = (32,32,5)
    batch_size = hp.Int('batch_size', min_value = 16, max_value = 128, step=8)
    lr = hp.Choice('learning_rate', values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
    alp = hp.Float('alpha', min_value = 0, max_value = 0.25)
    neur1 = hp.Int('neurons1', min_value = 64, max_value = 512, step=32)
    neur2 = hp.Int('neurons2', min_value = 64, max_value = 512, step=32)
    neur3 = hp.Int('neurons3', min_value = 64, max_value = 512, step=32)
    drop = hp.Float('dropout_rate', min_value = 0, max_value = 0.5)

    optimizer = keras.optimizers.Adam(lr)            

    inp = layers.Input(input_shape)
    x = inp

    base_model = tf.keras.applications.ResNet50(include_top=False, weights=None,\
        input_shape=input_shape)
    base_model.trainabe = True
    x = base_model(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(neur1,activation=layers.LeakyReLU(alpha=alp))(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(neur2,activation=layers.LeakyReLU(alpha=alp))(x)
    x = layers.Dropout(drop)(x)
    x = layers.Dense(neur3,activation=layers.LeakyReLU(alpha=alp))(x)
    x = layers.Dropout(drop)(x)
    out = layers.Dense(1)(x)
   
    model = tf.keras.models.Model(inp, out)

    model.compile(loss=tf.keras.losses.MSE,\
        optimizer=optimizer)

    return model
