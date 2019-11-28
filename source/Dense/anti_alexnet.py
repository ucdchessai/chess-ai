from   tensorflow import keras

def get_new():
    """
    Returns a new artificial neural network model with randomly initialized
    weights.
    """
    activation_hidden = __get_leaky_relu()

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
        optimizer=keras.optimizers.Adam(),
        loss='categorical_crossentropy'
    )

    return model


def load_model(filename):
    """
    Instantiates and returns the Anti-Alex Net saved in a file with the name
    ``filename``.
    """

    return keras.models.load_model(filename, custom_objects={'LeakyReLU':
            __get_leaky_relu()})


def get_dense(hidden_layer_count=5, layer_node_count=16,
        activation_hidden='relu'):
    """
    Returns a densely connected network with the specified parameters.
    """

    if (activation_hidden == 'leakyrelu'):
        activation_hidden = __get_leaky_relu()

    layer_input_count = 64
    specification = {
            'type': 'Dense',
            'units': layer_node_count,
            'activation': activation_hidden,
            'use_bias': True,
            'kernel_initializer': 'glorot_uniform',
            'bias_initializer': 'zeros',
            'kernel_regularizer': None,
            'bias_regularizer': None,
            'activity_regularizer': None,
            'kernel_constraint': None,
            'bias_constraint': None,
            'input_dim': layer_input_count
    }

    model = keras.models.Sequential()

    # Add hidden layers.
    for i in range(hidden_layer_count):
        specification['input_dim'] = layer_input_count
        hidden_layer = __instantiate_dense(specification)
        layer_input_count = layer_node_count
        model.add(hidden_layer)

    # Add output layer.
    specification['units'] = 2
    specification['activation'] = 'softmax'
    specification['input_dim'] = layer_input_count
    output_layer = __instantiate_dense(specification)
    model.add(output_layer)

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss='categorical_crossentropy'
    )

    return model


def __get_leaky_relu():
    """
    Returns a leaky ReLU loss function with a leaky slope of 0.1.
    """

    return keras.layers.LeakyReLU(0.1)


def __instantiate_layer(specification):
    """
    Returns a layer matching the provided ``specification``.
    """

    instantiators = {
        'Dense': __instantiate_dense
    }
    return instantiators[specification['type']](specification)


def __instantiate_dense(specification):
    """
    Returns a ``Dense``` layer matching the provided ``specification``.
    """

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
