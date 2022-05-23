"""Contains class and methods for encoding hyperparameters
into the genome as well as utilities for loading configurations from file"""
import copy
import json
import os
import random
import tensorflow as tf
import yaml
from configs.__init__ import CONFIG_DIR


class EvoEncoder(json.JSONEncoder):
    """JSON encoders for objects in this module"""

    def default(self, obj):
        if isinstance(obj, EvoConfig):
            return obj.serialize()
            # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


class EvoConfig:
    """A class for representing hyperparameters in the genome,
    as well as capturing evolution configuration parameters"""

    def __init__(self):
        self.config = {}
        self.loss = None
        self.opt = None
        self.callbacks = None
        self.input_dtype = None

    def __str__(self):
        return str(self.config)

    def _update_config(self, config: dict):
        for key, value in config.items():
            self.config[key] = value
            if key == 'opt':
                self.opt = self.setup_optimizer(self.config)
            elif key == 'loss':
                self.loss = self.setup_loss(self.config)
            elif key == 'early_stopping':
                self.callbacks = self.setup_callbacks(self.config)
            elif key == 'input_dtype':
                self._sort_out_dtype()
            else:
                pass

    def serialize(self) -> dict:
        """Converts object into serializable dict"""
        return copy.deepcopy(self.config)

    def save(self, filename: str):
        """Saves object to file as json"""
        with open(filename, 'w+', encoding='latin-1') as file:
            json.dump(self, fp=file, cls=EvoEncoder)

    @staticmethod
    def load(filename: str):
        """Builds an object from a json file"""
        with open(filename, 'r', encoding='latin-1') as file:
            config_dict = json.load(file)
            loaded_evo_config = EvoConfig.deserialize(config_dict)
            return loaded_evo_config

    @staticmethod
    def deserialize(config_dict: dict):
        """Rebuild object from a dict"""
        new_evo_config = EvoConfig()
        new_evo_config._update_config(config_dict)
        return new_evo_config

    def get_default_config(self):
        """Load the default evolutionary configuration from file"""

        path = os.path.join(CONFIG_DIR, 'default_config.yaml')
        with open(path, encoding='latin-1') as file:
            default_config = yaml.safe_load(file)
        self.config = default_config
        self.opt = self.setup_optimizer(self.config)
        self.loss = self.setup_loss(self.config)
        self.callbacks = self.setup_callbacks(self.config)
        self._sort_out_dtype()

    @staticmethod
    def load_config(file_name: str) -> dict:
        """Loads a config from a yaml file
        Args:
            file_name: path to yaml config file
        """
        with open(file_name, encoding='latin-1') as file:
            loaded_config = yaml.safe_load(file)
        return loaded_config

    @staticmethod
    def setup_optimizer(passed_config: dict):
        """Builds optimizer from a config

           Args:
               passed_config: a configuration dictionary containing optimizer
                   information
               """

        opt_dict = passed_config['opt']
        optimizer = opt_dict['optimizer']
        learning_rate = opt_dict['learning_rate']
        passed_config['learning_rate'] = learning_rate

        if passed_config['backend'] == 'tf':
            if optimizer == 'adam':
                configured_opt = tf.keras.optimizers.Adam(learning_rate)
            else:
                raise ValueError("Unsupported optimizer")
        else:
            raise ValueError("Unsupported backend")
        return configured_opt

    @staticmethod
    def setup_loss(passed_config: dict):
        """Builds loss from a config

           Args:
               passed_config: a configuration dictionary containing loss
                   information
               """

        loss_str = passed_config['loss']
        if passed_config['backend'] == 'tf':
            if loss_str == 'SparseCategoricalCrossentropy':
                configured_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            elif loss_str == 'MeanSquaredError':
                configured_loss = tf.keras.losses.MeanSquaredError()
            elif loss_str == 'MeanAbsoluteError':
                configured_loss = tf.keras.losses.MeanAbsoluteError()
            elif loss_str == 'BinaryCrossentropy':
                configured_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            else:
                raise ValueError("Unsupported loss function")
        else:
            raise ValueError("Unsupported backend")
        return configured_loss

    @staticmethod
    def setup_callbacks(passed_config: dict) -> list:
        """Builds callbacks from a config

        Args:
            passed_config: a configuration dictionary containing callback
                information
            """
        callback_list = []

        early_stopping_dict = passed_config['early_stopping']
        use_early = early_stopping_dict['use_early']

        if use_early:
            monitor = early_stopping_dict['monitor']
            patience = early_stopping_dict['patience']
            min_delta = early_stopping_dict['min_delta']

            if passed_config['backend'] == 'tf':
                early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                                  patience=patience,
                                                                  min_delta=min_delta)
            else:
                raise ValueError("Unsupported backend")

            callback_list.append(early_stopping)
        return callback_list

    def setup_user_config(self, user_config):
        """Updates the configuration based on user input.

        Args:
            user_config: either a dictionary or the path to a
            yaml file that contains configuration information
        """
        if user_config is None:
            return

        if isinstance(user_config, dict):
            new_config = user_config
        elif isinstance(user_config, str):
            new_config = self.load_config(user_config)
        else:
            raise ValueError("Invalid config type")

        self._update_config(new_config)

    def clone(self):
        """Returns: a deep copy of the current class instance"""
        new_evo_config = EvoConfig()
        config_copy = copy.deepcopy(self.config)
        new_evo_config._update_config(config_copy)
        return new_evo_config

    def __deepcopy__(self, memodict={}):
        return self.clone()

    def mutate(self):
        """Handles mutation of the hyperparameters.
        Filters all hyperparams for just those that can be mutated, then
        chooses one of those mutatable params at random and
        mutates it"""

        param_to_mutate = random.choice(list(self.get_mutatable_params().keys()))

        if param_to_mutate in ('cx', 'm_mut', 'm_del', 'm_insert', 'hyper_cx', 'hyper_mut'):
            self.config[param_to_mutate] = random.random()

        elif param_to_mutate == 'learning_rate':
            learning_rate = random.choice(self.config['learning_rates'])
            learning_rate *= random.randint(1, 9)
            self.config['learning_rate'] = learning_rate
            self.opt = self.setup_optimizer(self.config)

        elif param_to_mutate == 'opt':
            pass
        else:
            raise ValueError("Unsupported parameter to mutate: " + str(param_to_mutate))

        # print("Mutated: " + param_to_mutate + " to: " + str(self.config[param_to_mutate]))

    def get_mutatable_params(self) -> dict:
        """Filters all possible hyperparameters for only those which are
        valid to mutate

         Returns:
             A dict of valid params to mutate
        """

        return {k: v for (k, v) in self.config.items() if k in self.config['mutatable_keys']}

    def random_regularizer(self) -> list:
        """Generates a random kernel regularizer based on the
        relevant parameters from the config dictionary"""
        config = self.config
        random_reg = random.choice(config['regularizer_types'])
        l1 = random.choice(config['regulizer_factor'])
        l2 = random.choice(config['regulizer_factor'])

        return [random_reg, l1, l2]

    @staticmethod
    def build_regularizer(regularizer_list: list) -> tf.keras.regularizers.Regularizer:
        """Builds regularizer"""
        regularizer = regularizer_list[0]
        l1 = regularizer_list[1]
        l2 = regularizer_list[2]

        if regularizer in [None, 'None']:
            return None
        elif regularizer == "L1L2":
            return tf.keras.regularizers.L1L2(l1=l1, l2=l2)
        elif regularizer == "L1":
            return tf.keras.regularizers.L1(l1=l1)
        elif regularizer == "L2":
            return tf.keras.regularizers.L2(l2=l2)
        else:
            raise AttributeError("Unsupported kernel regularizer: " + str(regularizer))

    def _sort_out_dtype(self):
        dtype = self.config["input_dtype"]
        if dtype == "None":
            self.input_dtype = None
        elif dtype == "tf.string":
            self.input_dtype = tf.string
        else:
            raise ValueError("Unsupported input dtype")


def cross_over(some_hp: EvoConfig, other_hp: EvoConfig):
    """Cross over operator for hyperparameters. Filters for valid parameters
    to mutate, then executes uniform crossover

    Args:
        some_hp: A hyperparam object
        other_hp: Another hyperparam object
    """
    for key in some_hp.get_mutatable_params().keys():
        if random.random() < some_hp.config['hyper_cx_uniform_prob']:
            some_hp_param = some_hp.config[key]
            other_hp_param = other_hp.config[key]
            some_hp.config[key] = other_hp_param
            other_hp.config[key] = some_hp_param


master_config = EvoConfig()
master_config.get_default_config()
