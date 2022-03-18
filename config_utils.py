import yaml
import tensorflow as tf


def get_default_config() -> dict:
    with open('config.yaml') as file:
        default_config = yaml.safe_load(file)
    return default_config


def load_config(file_name: str) -> dict:
    with open(file_name) as file:
        loaded_config = yaml.safe_load(file)
    return loaded_config


def setup_optimizer(config: dict):
    opt_dict = config['opt']
    optimizer = opt_dict['optimizer']
    learning_rate = opt_dict['learning_rate']

    if config['backend'] == 'tf':
        if optimizer == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate)
        else:
            raise ValueError("Unsupported optimizer")
    else:
        raise ValueError("Unsupported backend")
    return opt


def setup_loss(config: dict):
    loss_str = config['loss']
    if config['backend'] == 'tf':
        if loss_str == 'SparseCategoricalCrossentropy':
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            raise ValueError("Unsupported loss function")
    else:
        raise ValueError("Unsupported backend")
    return loss


def setup_callbacks(config: dict) -> list:
    callbacks = []

    early_stopping_dict = config['early_stopping']
    use_early = early_stopping_dict['use_early']

    if use_early:
        monitor = early_stopping_dict['monitor']
        patience = early_stopping_dict['patience']
        min_delta = early_stopping_dict['min_delta']

        if config['backend'] == 'tf':
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                              patience=patience, min_delta=min_delta)
        else:
            raise ValueError("Unsupported backend")

        callbacks.append(early_stopping)
    return callbacks


def _setup_user_config(existing_config: dict, user_config):
    if user_config is None:
        return

    if type(user_config) is dict:
        new_config = user_config
    elif type(user_config) is str:
        new_config = load_config(user_config)
    else:
        raise ValueError("Invalid config type")

    for key, value in new_config.items():
        existing_config[key] = value


config = get_default_config()
loss = setup_loss(config)
opt = setup_optimizer(config)
callbacks = setup_callbacks(config)


def setup_config(user_config):
    _setup_user_config(config, user_config)
    global loss
    global opt
    global callbacks
    loss = setup_loss(config)
    opt = setup_optimizer(config)
    callbacks = setup_callbacks(config)
