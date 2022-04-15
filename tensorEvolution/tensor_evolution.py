"""This module contains the main evolution loop"""

import json

import random
import numpy
import ray
import tensorflow as tf
from deap import creator, base, tools
from matplotlib import pyplot as plt

from tensorEvolution import tensor_encoder
from tensorEvolution.ActorPoolExtension import ActorPoolExtension
from tensorEvolution import tensor_network
from tensorEvolution.nodes import node_utils
from tensorEvolution import evo_config


# noinspection PyCallingNonCallable
@ray.remote(num_cpus=evo_config.master_config.config['remote_actor_cpus'],
            num_gpus=evo_config.master_config.config['remote_actor_gpus'])
class RemoteEvoActor:
    """This class is a Ray remote actor.
    This actor performs evaluation on individuals in the population."""

    def __init__(self, data):
        """initialize an actor

            Args:
                data: data which will be passed to the model.fit method for training. Must be
                    a tuple of form (x_train, y_train, x_test, y_test)
            """

        self.data = data

    def eval(self, individual: list):
        """Remote evaluation function, evaluates and individual's fitness.
        The individual is evaluated by compiling and training a model based on
        the individual's genome. The test accuracy is used as the fitness (minus
        a penalty for model complexity)

         Args:
             individual: An individual from the population
             """

        config = individual[0]
        x_train, y_train, x_test, y_test = self.data
        tensor_net = individual[1]
        model = tensor_net.build_model()
        model.compile(loss=config.loss,
                      optimizer=config.opt,
                      metrics=config.config['metrics'])

        batch_size = config.config['batch_size']
        if batch_size == 'None':
            batch_size = None

        model.fit(x_train, y_train, epochs=config.config['max_fit_epochs'],
                  callbacks=config.callbacks, verbose=config.config['verbose'],
                  batch_size=batch_size)
        _, test_acc = model.evaluate(x_test, y_test)

        length = len(individual[1].get_middle_nodes())
        complexity_penalty = config.config['complexity_penalty']
        penalty = complexity_penalty * length
        # noinspection PyRedundantParentheses
        return (test_acc - penalty,)


# def cx_chain(ind1, ind2):
#     tn = ind1[1]
#     other_tn = ind2[1]
#     tensor_network.cx_chain(tn, other_tn)
#     return ind1, ind2


class EvolutionWorker:
    """Main class which drives evolution. Contains
    relevant methods for setting itself up, building a population,
    executing mutation and crossover, etc."""

    def __init__(self):
        self.record = None
        self.logbook = None
        self.stats = None
        self.pop = None
        self.pool = None
        self.toolbox = base.Toolbox()
        self._setup_stats()
        self._setup_log()
        self.master_config = evo_config.master_config
        self.preprocessing_layers = None
        self.preprocessing_layers_length = 0
        self._setup_creator()

    def update_master_config(self, config):
        """Updates the configuration based on user input.

                Args:
                    config: either a dictionary or a yaml file
                    that contains configuration information
        """
        self.master_config.setup_user_config(config)

    def _setup_toolbox(self):
        # pylint: disable=E1101
        self.toolbox.register("individual", self._initialize_ind,
                              input_shapes=self.master_config.config['input_shapes'],
                              num_outputs=self.master_config.config['num_outputs'])
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self._cx_single_node)
        self.toolbox.register("cx_hyper", self._cx_hyper)
        self.toolbox.register("mutate_insert", self._mutate_insert)
        self.toolbox.register("mutate_delete", self._mutate_delete)
        self.toolbox.register("mutate_mutate", self._mutate_mutate)
        self.toolbox.register("mutate_hyper", self._mutate_hyper)
        self.toolbox.register("select", tools.selTournament,
                              tournsize=self.master_config.config['t_size'])

    def _setup_stats(self):
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", numpy.mean)
        # stats.register("std", numpy.std)
        self.stats.register("min", numpy.min)
        self.stats.register("max", numpy.max)

    def _setup_log(self):
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "avg", "max", "min"

    def plot(self):
        """Basic method for plotting fitness at each generation"""
        gen, avg = self.logbook.select("gen"), self.logbook.select("avg")
        _, axis = plt.subplots()
        axis.plot(gen, avg, "g-", label="Avg Fitness")
        axis.set_xlabel("Generation")
        axis.set_ylabel("Fitness", color="b")
        plt.show()

    def _initialize_ind(self, input_shapes, num_outputs) -> list:
        # pylint: disable=E1101
        ind = creator.Individual()
        individual_config = self.master_config.clone()

        tensor_net = tensor_network.TensorNetwork(input_shapes, num_outputs,
                                                  preprocessing_layers=self.preprocessing_layers)
        ind.append(individual_config)
        ind.append(tensor_net)
        return ind

    @staticmethod
    def _eval_remote(actor: RemoteEvoActor, individual: list):
        individual = list(individual)
        return actor.eval.remote(individual)

    @staticmethod
    def _evaluate(individual: list, data: tuple):
        x_train, y_train, x_test, y_test = data
        config = individual[0]
        tensor_net = individual[1]

        model = tensor_net.build_model()

        model.compile(loss=config.loss, optimizer=config.opt,
                      metrics=config.config['metrics'])

        if config.config['global_cache_training']:
            tensor_net.store_weights(model, direction_into_tn=False)

        batch_size = config.config['batch_size']
        if batch_size == 'None':
            batch_size = None

        model.fit(x_train, y_train, epochs=config.config['max_fit_epochs'],
                  callbacks=config.callbacks, verbose=config.config['verbose'],
                  batch_size=batch_size)
        _, test_acc = model.evaluate(x_test, y_test)

        if config.config['global_cache_training']:
            tensor_net.store_weights(model, direction_into_tn=True)

        length = len(individual[1].get_middle_nodes())
        penalty = config.config['complexity_penalty'] * length
        # noinspection PyRedundantParentheses
        return (test_acc - penalty,)

    @staticmethod
    def _is_too_big(individual) -> bool:
        config = individual[0]
        tensor_net = individual[1]

        size = len(tensor_net.get_middle_nodes())
        max_size = config.config['max_network_size']

        if size >= max_size:
            return True
        return False

    @staticmethod
    def _mutate_insert(individual: list):
        if EvolutionWorker._is_too_big(individual):
            return

        config = individual[0]
        tensor_net = individual[1]

        position = random.randint(0, len(tensor_net.get_valid_insert_positions()) - 1)
        node_type = random.choice(config.config['valid_node_types'])
        node = node_utils.create(node_type)
        tensor_net.insert_node(node, position)
        # noinspection PyRedundantParentheses
        return (individual,)

    @staticmethod
    def _mutate_mutate(individual: list):
        tensor_net = individual[1]
        length = len(tensor_net.get_mutatable_nodes())

        if length == 0:  # nothing to mutate
            # noinspection PyRedundantParentheses
            return (individual,)

        position = random.randint(0, length - 1)
        tensor_net.mutate_node(position)
        # noinspection PyRedundantParentheses
        return (individual,)

    @staticmethod
    def _mutate_delete(individual: list):
        tensor_net = individual[1]
        length = len(tensor_net.get_middle_nodes())

        if length == 0:  # nothing to delete
            # noinspection PyRedundantParentheses
            return (individual,)

        position = random.randint(0, length - 1)
        node_id = list(tensor_net.get_middle_nodes().keys())[position]
        tensor_net.delete_node(node_id=node_id)
        # noinspection PyRedundantParentheses
        return (individual,)

    @staticmethod
    def _mutate_hyper(individual: list):
        ind_hyper_params = individual[0]
        ind_hyper_params.mutate()
        # noinspection PyRedundantParentheses
        return (individual,)

    @staticmethod
    def _cx_hyper(ind1, ind2):
        hyper1 = ind1[0]
        hyper2 = ind2[0]
        evo_config.cross_over(hyper1, hyper2)
        return ind1, ind2

    @staticmethod
    def _cx_single_node(ind1, ind2):
        tensor_net = ind1[1]
        other_net = ind2[1]
        tensor_network.cx_single_node(tensor_net, other_net)
        return ind1, ind2

    def _setup_creator(self):
        # pylint: disable=E1101
        direction = self.master_config.config['direction']

        if direction == 'max':
            creator.create("Fitness", base.Fitness, weights=(1.0,))
        else:
            creator.create("Fitness", base.Fitness, weights=(-1.0,))

        creator.create("Individual", list, fitness=creator.Fitness)

    def evolve(self, data):
        """Main evolution method. Call to begin evolution.

        Args:
                data: data which will be passed to the model.fit method for training. Must be
                    a tuple of form (x_train, y_train, x_test, y_test)
        """
        # pylint: disable=E1101

        if not self.master_config.config['remote']:
            self.toolbox.register("evaluate", self._evaluate, data=data)
        else:
            ray.init()
            actors = []
            for _ in range(self.master_config.config['remote_actors']):
                actor = RemoteEvoActor.remote(data)
                actors.append(actor)

            self.pool = ActorPoolExtension(actors)
            self.toolbox.register("evaluate", self._eval_remote)

        self._evolve()

    def _crossover_on_population(self, offspring: list):
        # pylint: disable=E1101

        # Apply crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            cx_min = min((child1[0]).config['cx'], (child2[0]).config['cx'])
            if random.random() < cx_min:
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

            if self.master_config.config['evolve_hyperparams']:
                hyper_cx_min = min((child1[0]).config['hyper_cx'], (child2[0]).config['hyper_cx'])
                if random.random() < hyper_cx_min:
                    self.toolbox.cx_hyper(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

    def _mutation_on_population(self, offspring: list):
        # pylint: disable=E1101

        # apply mutation
        for mutant in offspring:
            mutant_hyper = mutant[0]

            if random.random() < mutant_hyper.config['m_insert']:
                self.toolbox.mutate_insert(mutant)
                del mutant.fitness.values

            if random.random() < mutant_hyper.config['m_del']:
                self.toolbox.mutate_delete(mutant)
                del mutant.fitness.values

            if random.random() < mutant_hyper.config['m_mut']:
                self.toolbox.mutate_mutate(mutant)
                del mutant.fitness.values

            if self.master_config.config['evolve_hyperparams']:
                if random.random() < mutant_hyper.config['hyper_mut']:
                    self.toolbox.mutate_hyper(mutant)
                    del mutant.fitness.values

    def _gen_bookkeeping(self, gen):
        self.record = self.stats.compile(self.pop)
        self.logbook.record(gen=gen, **self.record)
        print('\n')
        print(self.logbook.header)
        print(self.logbook.stream)
        print('\n')

    def _evolve(self):
        # pylint: disable=E1101

        # set up toolbox
        self._setup_toolbox()

        # build population
        if self.pop is None:
            self.pop = self.toolbox.population(n=self.master_config.config['pop_size'])
            self._save_preprocessing_layers()

        # Evaluate the entire population
        if self.master_config.config['remote']:
            fitnesses = self.pool.map_ordered_return_all(self.toolbox.evaluate, self.pop)
        else:
            fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        # Begin main evolution loop
        for gen in range(self.master_config.config['ngen']):
            # Select the next generation individuals
            offspring = self.toolbox.select(self.pop, len(self.pop))

            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # apply operators
            self._crossover_on_population(offspring)
            self._mutation_on_population(offspring)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            if self.master_config.config['remote']:
                fitnesses = self.pool.map_ordered_return_all(self.toolbox.evaluate, invalid_ind)
            else:
                fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            self.pop[:] = offspring

            # Record and print info about this generation
            self._gen_bookkeeping(gen)

            # save data to disk
            self._save_every(gen)

        print("-- End of (successful) evolution --")

        best_ind = tools.selBest(self.pop, 1)[0]
        print("Best individual is %s" % best_ind.fitness.values)
        best_ind[1].build_model().summary()
        self.save(self.master_config.config['save_pop_filepath'])
        self.plot()

    def get_best_individual(self):
        """Returns the best individual in the population"""
        return tools.selBest(self.pop, 1)[0]

    def _save_every(self, gen):
        if (gen % self.master_config.config['save_pop_every']) == 0:
            self.save(self.master_config.config['save_pop_filepath'])

    def serialize(self) -> dict:
        """Currently just saves the population and master config"""
        serial_dict = {}
        serial_dict['pop'] = self.serialize_pop()
        serial_dict['master_config'] = self.master_config.serialize()
        return serial_dict

    def save(self, filename: str):
        """Saves object to file as json. Currently, just save population and master config
        Args:
            filename: full filepath to save to
        """
        with open(filename, 'w+', encoding='latin-1') as file:
            json.dump(self, fp=file, cls=tensor_encoder.TensorEncoder)

    @staticmethod
    def load(filename: str):
        """Builds an object from a json file
        Args:
            filename: full path to saved json file
        """

        with open(filename, 'r', encoding='latin-1') as file:
            serial_dict = json.load(file)
            loaded_evo_worker = EvolutionWorker.deserialize(serial_dict)
            return loaded_evo_worker

    @staticmethod
    def deserialize(serial_dict: dict):
        """Rebuild object from a dict"""
        new_worker = EvolutionWorker()
        new_worker.master_config = evo_config.EvoConfig.deserialize(serial_dict['master_config'])
        new_worker._load_preprocessing_layers()
        new_worker.pop = EvolutionWorker.deserialize_pop(serial_dict['pop'],
                                                         new_worker.preprocessing_layers)
        return new_worker

    def serialize_pop(self) -> list:
        """Converts population to serializable form"""
        pop = []
        for individual in self.pop:
            pop.append(self.serialize_individual(individual))
        return pop

    @staticmethod
    def serialize_individual(individual) -> list:
        """Converts an individual of the population to serial form"""
        serial_individual = []
        serial_individual.append(individual[0].serialize())
        serial_individual.append(individual[1].serialize())
        serial_individual.append(list(individual.fitness.values))
        return serial_individual

    @staticmethod
    def deserialize_individual(serial_individual: list, preprocessing_layers=None):
        """Rebuilds an individual from serial form"""
        # pylint: disable=E1101
        new_ind = creator.Individual()
        individual_config = evo_config.EvoConfig.deserialize(serial_individual[0])
        tensor_net = tensor_network.TensorNetwork.deserialize(serial_individual[1])
        if preprocessing_layers is not None:
            tensor_net.load_preprocessing_layers(preprocessing_layers)
        new_ind.append(individual_config)
        new_ind.append(tensor_net)
        new_ind.fitness.values = tuple(serial_individual[2])
        return new_ind

    @staticmethod
    def deserialize_pop(serial_pop: list, preprocessing_layers=None) -> list:
        """Rebuilds population from serial form"""
        new_pop = []
        for serial_individual in serial_pop:
            new_pop.append(EvolutionWorker.deserialize_individual(serial_individual,
                                                                  preprocessing_layers))
        return new_pop

    def setup_preprocessing(self, preprocessing_layers: list):
        """Use this method to set preprocessing layers on the input
        Args:
            preprocessing_layers: needs to be a list of lists,or the None value type.
            Each position in the outermost list is a set of preprocessing layers to be
            applied to genomes. The None value represents no preprocessing.
            When a new individual of the population is created, a set of preprocessing
            layers will be chosen randomly from the list (include None as a list member if
            you want the option of no preprocessing).
            The layers need to be ready to go (so run any adapting beforehand)"""
        self.preprocessing_layers = preprocessing_layers
        self.preprocessing_layers_length = len(preprocessing_layers)
        self._save_preprocessing_layers()

    def _save_preprocessing_layers(self):
        if self.preprocessing_layers is None:
            return

        input_shapes = self.master_config.config['input_shapes']
        num_outputs = self.master_config.config['num_outputs']

        inputs = tf.keras.Input(shape=tuple(input_shapes[0]))

        layers_so_far = inputs
        for pre_layer in self.preprocessing_layers:
            layers_so_far = pre_layer(layers_so_far)

        outputs = tf.keras.layers.Dense(num_outputs[0])(layers_so_far)
        model = tf.keras.Model(inputs, outputs)
        preprocessing_save_path = evo_config.master_config.config['preprocessing_save_path']
        model.save(preprocessing_save_path)

    def _load_preprocessing_layers(self):
        if self.preprocessing_layers_length == 0:
            return

        preprocessing_save_path = evo_config.master_config.config['preprocessing_save_path']
        model = tf.keras.models.load_model(preprocessing_save_path)

        prelayers = []
        for i in range(1, (self.preprocessing_layers_length+ 1)):
            prelayers.append(model.layers[i])

        self.preprocessing_layers = prelayers
