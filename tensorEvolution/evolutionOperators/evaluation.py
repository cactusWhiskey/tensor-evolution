"""Evaluation of an individual in the population"""

import ray

from tensorEvolution import util, evo_config


# # noinspection PyCallingNonCallable
# @ray.remote(num_cpus=evo_config.master_config.config['remote_actor_cpus'],
#             num_gpus=evo_config.master_config.config['remote_actor_gpus'])
# class RemoteEvoActor:
#     """This class is a Ray remote actor.
#     This actor performs evaluation on individuals in the population remotely."""
#
#     def __init__(self, data):
#         """initialize an actor
#
#             Args:
#                 data: data which will be passed to the model.fit method for training. Must be
#                     a tuple of form (x_train, y_train, x_test, y_test)
#             """
#
#         self.data = data
#
#     def eval(self, indexed_individual: tuple):
#         """Remote evaluation function, evaluates and individual's fitness.
#         The individual is evaluated by compiling and training a model based on
#         the individual's genome. The test accuracy is used as the fitness
#          Args:
#              indexed_individual: a tuple of form (index, individual),
#         where individual is an individual in the population (individuals are a list at this point)
#              """
#
#         return evaluate(indexed_individual, self.data)


def evaluate(indexed_individual: tuple, data) -> tuple:
    """Evaluate a single individual.
    Args:
        indexed_individual: a tuple of form (index, individual),
        where individual is an individual in the population (individuals inherit from list)
        data: data to be used for training and testing.
        In the form: (x_train, y_train, x_test, y_test)
        :return a tuple of form (fitness, Tensornet dict, index)
        """

    if not isinstance(data, tuple):
        data = ray.get(data)

    # unpack data tuple
    x_train, y_train, x_test, y_test = data
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

    # check if we should be trying retrieve weights
    if config.config['global_cache_training']:
        tensor_net.get_weights(model)

    # compute this tensornet's complexity (needed for the selection process later on)
    tensor_net.complexity = util.compute_complexity(model)

    # check what the batch size should be
    batch_size = config.config['batch_size']
    if batch_size == 'None':
        batch_size = None

    # fit the model
    model.fit(x_train, y_train, epochs=config.config['max_fit_epochs'],
              callbacks=config.callbacks, verbose=config.config['verbose'][0],
              batch_size=batch_size)
    # evaluate the model
    test_loss, test_metric = model.evaluate(x_test, y_test, verbose=config.config['verbose'][1])

    if config.config['global_cache_training']:
        tensor_net.set_weights(model)

    # del model

    # noinspection PyRedundantParentheses
    return ((test_metric,), tensor_net.__dict__, index)


# def eval_remote(actor: RemoteEvoActor, indexed_individual: tuple):
#     """evaluation function for Ray remote actors. Used to map an actor to an individual.
#     Args:
#         actor: Ray remote actor
#         indexed_individual: a tuple of form (index, individual),
#         where individual is an individual in the population (individuals inherit from list)
#     :return """
#     index, individual = indexed_individual  # unpack tuple
#     individual = list(individual)  # convert individual to a list for serialization
#     # (this loses the stored fitness data, but we don't need it right now)
#
#     # repack tuple
#     indexed_individual = (index, individual)
#     # call remote function
#     return actor.eval.remote(indexed_individual)


# noinspection PyCallingNonCallable
@ray.remote(num_cpus=evo_config.master_config.config['ray_cpus_task'],
            num_gpus=evo_config.master_config.config['ray_gpus_task'],
            memory=evo_config.master_config.config['ray_mem_task'])
def eval_ray_task(indexed_individual: tuple, data_id):
    return evaluate(indexed_individual, data_id)
