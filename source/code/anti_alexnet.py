import tensorflow as tf
from   tensorflow import keras

def get_new():
    """
    Returns a new artificial neural network model with randomly initialized
    weights.
    """
    activation_hidden = keras.layers.LeakyReLU(0.1)

    specifications = [
        {
            'type': 'Dense',
            'units': 64,
            'activation': activation_hidden,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'bias_constraint': None,
            'input_dim': 64
        },
        {
            'type': 'Dense',
            'units': 32,
            'activation': activation_hidden,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'bias_constraint': None,
            'input_dim': 64
        },
        {
            'type': 'Dense',
            'units': 16,
            'activation': activation_hidden,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'bias_constraint': None,
            'input_dim': 32
        },
        {
            'type': 'Dense',
            'units': 8,
            'activation': activation_hidden,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'bias_constraint': None,
            'input_dim': 16
        },
        {
            'type': 'Dense',
            'units': 4,
            'activation': activation_hidden,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'bias_constraint': None,
            'input_dim': 8
        },
        {
            'type': 'Dense',
            'units': 2,
            'activation': 'softmax',
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'bias_constraint': None,
            'input_dim': 8
        }
    ]
    model = keras.models.Sequential()

    for s in specifications:
        model.add(__instantiate_layer(s))

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=.1),
        #loss='mean_squared_error'
        loss='categorical_crossentropy'
    )

    return model


def load_model(filename):
    """
    Instantiates and returns the Anti-Alex Net saved in a file with the name
    ``filename``.
    """

    return keras.models.load_model(filename, custom_objects={'LeakyReLU':
            keras.layers.LeakyReLU(0.1)})


def __instantiate_layer(specification):
    instantiators = {
        'Dense': __instantiate_dense
    }
    return instantiators[specification['type']](specification)


def __instantiate_dense(specification):
    s = specification
    return keras.layers.Dense(
        units=s['units'],
        activation=s['activation'],
        use_bias=s['use_bias'],
        kernel_initializer=s['kernel_initializer'],
        bias_initializer=s['bias_initializer'],
        kernel_regularizer=s['kernel_regularizer'],
        bias_regularizer=s['bias_regularizer'],
        activity_regularizer=s['activity_regularizer'],
        kernel_constraint=s['kernel_constraint'],
        bias_constraint=s['bias_constraint'],
        input_dim=s['input_dim'])