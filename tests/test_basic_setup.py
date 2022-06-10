"""Simple test to verify tensor_evolution module imports, worker can be initialized, and a simple example run"""
import numpy as np
from tensorEvolution import tensor_evolution, evo_config


def test_basic_setup():
    # build custom config
    custom_config = {}
    custom_config['input_shapes'] = [[1]]
    custom_config['num_outputs'] = [1]
    custom_config['pop_size'] = 5
    custom_config['remote_mode'] = 'ray_remote'
    custom_config['loss'] = 'MeanAbsoluteError'
    custom_config['max_fit_epochs'] = 2
    custom_config['metrics'] = ['mean_absolute_error']
    custom_config['ngen'] = 3
    custom_config['direction'] = 'min'
    custom_config['batch_data_per_gen'] = True
    custom_config['gen_batch_datatype'] = 'numpy_data'
    custom_config['batch_data_train_size'] = 10
    custom_config['batch_data_val_size'] = 10

    # set the custom config
    evo_config.master_config.setup_user_config(custom_config)

    assert evo_config.master_config.config['pop_size'] == 5
    assert evo_config.master_config.config['remote_mode'] == 'ray_remote'

    train_features = np.arange(0, 15)
    train_labels = np.arange(0, 15)
    val_features = np.arange(0, 15)
    val_labels = np.arange(0, 15)

    data = (train_features, train_labels, val_features, val_labels)

    worker = tensor_evolution.EvolutionWorker()
    assert worker is not None

    # run the evolution
    worker.evolve(data=data)

