"""Evaluation of an individual in the population"""

import ray
from tensorEvolution import util, evo_config


def evaluate(indexed_individual: tuple, data) -> tuple:
    """Evaluate a single individual.
    Args:
        indexed_individual: a tuple of form (index, individual),
        where individual is an individual in the population (individuals inherit from list)
        data: data to be used for training and testing.

        :return a tuple of form (fitness, Tensornet dict, index)
        """
    using_generator = False

    # get the data from the object store if we were passed an object ref
    if isinstance(data, ray._raylet.ObjectRef):
        data = ray.get(data)

    # check data length (to determine how it's packed)
    if len(data) == 2:
        # train and validation data passed as generators or tf datasets
        using_generator = True
        train_data = data[0]
        validation_data = data[1]
    elif len(data) == 4:
        train_data = data[0]
        train_labels = data[1]
        validation_data = data[2]
        validation_labels = data[3]
    else:
        raise ValueError("unsupported data format")

    # unpack indexed individual
    index, individual = indexed_individual
    # config is always in position 0 of the genome
    config = individual[0]
    # tensornet is always in position 1 of the genome
    tensor_net = individual[1]

    # build and compile model
    model = tensor_net.build_model()
    model.compile(loss=config.loss, optimizer=config.opt,
                  metrics=config.config['metrics'])

    # # check if we should be trying retrieve weights
    # if config.config['global_cache_training']:
    #     tensor_net.get_weights(model)

    # compute this tensornet's complexity (needed for the selection process later on)
    tensor_net.complexity = util.compute_complexity(model)

    # check what the batch size should be
    batch_size = config.config['batch_size']
    if batch_size == 'None':
        batch_size = None

    if using_generator:
        # fit the model
        model.fit(train_data, epochs=config.config['max_fit_epochs'],
                  callbacks=config.callbacks, verbose=config.config['verbose'][0],
                  batch_size=batch_size)
        # evaluate the model
        test_loss, test_metric = model.evaluate(validation_data, verbose=config.config['verbose'][1])
    else:
        # fit the model
        model.fit(train_data, train_labels, epochs=config.config['max_fit_epochs'],
                  callbacks=config.callbacks, verbose=config.config['verbose'][0],
                  batch_size=batch_size)
        # evaluate the model
        test_loss, test_metric = model.evaluate(validation_data, validation_labels,
                                                verbose=config.config['verbose'][1])

    if config.config['global_cache_training']:
        tensor_net.set_weights(model)

    # noinspection PyRedundantParentheses
    return ((test_metric,), tensor_net.__dict__, index)


# noinspection PyCallingNonCallable
@ray.remote(num_cpus=evo_config.master_config.config['ray_cpus_task'],
            num_gpus=evo_config.master_config.config['ray_gpus_task'],
            memory=evo_config.master_config.config['ray_mem_task'])
def eval_ray_task(indexed_individual: tuple, data_id):
    return evaluate(indexed_individual, data_id)
