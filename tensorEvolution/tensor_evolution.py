"""This module contains the main evolution loop"""

import json
import random

import numpy
import ray
import tensorflow as tf
from deap import creator, base, tools
from matplotlib import pyplot as plt
from ray.util.actor_pool import ActorPool

from tensorEvolution import evo_config
from tensorEvolution import tensor_encoder
from tensorEvolution import tensor_network
from tensorEvolution.evolutionOperators import selection, crossover, mutation, evaluation
from tensorEvolution.evolutionOperators.evaluation import RemoteEvoActor


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
        self.toolbox.register("mate", crossover.cx_single_node)
        self.toolbox.register("cx_hyper", crossover.cx_hyper)
        self.toolbox.register("mutate_insert", mutation.mutate_insert)
        self.toolbox.register("mutate_delete", mutation.mutate_delete)
        self.toolbox.register("mutate_mutate", mutation.mutate_mutate)
        self.toolbox.register("mutate_hyper", mutation.mutate_hyper)
        self.toolbox.register("select", selection.double_tournament,
                              fitness_t_size=self.master_config.config['fitness_t_size'],
                              prob_sel_smallest=self.master_config.config['prob_sel_least_complex'],
                              do_fitness_first=True)

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

        # configure the evaluation function based on remote vs local execution
        if not self.master_config.config['remote']:
            self.toolbox.register("evaluate", evaluation.evaluate, data=data)
        else:
            ray.init()
            actors = []
            for _ in range(self.master_config.config['remote_actors']):
                actor = RemoteEvoActor.remote(data)
                actors.append(actor)

            self.pool = ActorPool(actors)
            self.toolbox.register("evaluate", evaluation.eval_remote)

        # now start the evolution loop
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

    def _evaluation_on_individuals(self, individuals):
        # pass the index with the individual to the evaluation function
        # the index just gets passed back with the results so we can
        # identify which result goes with which individual later
        indexes_with_inds = [(index, ind) for index, ind in enumerate(individuals)]

        # check if remote
        remote = self.master_config.config['remote']
        if remote:
            # use remote function
            eval_results = list(self.pool.map_unordered(self.toolbox.evaluate, indexes_with_inds))
        else:
            # use local function
            eval_results = list(map(self.toolbox.evaluate, indexes_with_inds))

        for result in eval_results:
            # fitness tuple is always in the zero position of the evaluation returned results
            fitness = result[0]
            # index is in the 2 position
            index = result[2]
            # get the individual this fitness result is for
            this_individual = individuals[index]
            # set the fitness
            this_individual.fitness.values = fitness
            if remote:
                # remote execution has its own copies of the individuals it evaluated,
                # so complexity and weight cache didn't get done on the local (master) copy of
                # the individuals. Need to do that now.

                # __dict__ from the remote individual is returned by the eval function at position 1
                tensor_dict_from_remote = result[1]
                tensor_net = this_individual[1]  # our local copy of this individual
                # copy the updated state over
                tensor_net.__dict__ = tensor_dict_from_remote

    def _gen_bookkeeping(self, gen):
        self.record = self.stats.compile(self.pop)
        self.logbook.record(gen=gen, **self.record)
        print('\n')
        print(self.logbook.header)
        print(self.logbook.stream)
        print('\n')

    def _evolve(self):
        # pylint: disable=E1101
        self._pre_evolution_tasks()
        self._main_evolution_loop()
        self._post_evolution_tasks()

    def _pre_evolution_tasks(self):
        # set up toolbox
        self._setup_toolbox()

        # build population if it wasn't loaded from disk
        if self.pop is None:
            self.pop = self.toolbox.population(n=self.master_config.config['pop_size'])
            self._save_preprocessing_layers()

        # compute initial fitness of population
        self._evaluation_on_individuals(self.pop)

    def _main_evolution_loop(self):
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
            self._evaluation_on_individuals(invalid_ind)

            # The population is entirely replaced by the offspring
            self.pop[:] = offspring

            # Record and print info about this generation
            self._gen_bookkeeping(gen)

            # save data to disk
            self._save_every(gen)

            print(f"Largest: {self._get_biggest_individual_size()}, "
                  f"Avg: {self._get_average_individual_size()}")

    def _post_evolution_tasks(self):
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
        for i in range(1, (self.preprocessing_layers_length + 1)):
            prelayers.append(model.layers[i])

        self.preprocessing_layers = prelayers

    def _get_biggest_individual_size(self):
        """For debugging use, returns member of population with the most nodes.
        Useful for checking if bloat control is working."""
        biggest = 0
        for individual in self.pop:
            tensor_net = individual[1]
            size = len(tensor_net.get_middle_nodes())
            biggest = max(biggest, size)
        return biggest

    def _get_average_individual_size(self):
        sum_sizes = 0
        for individual in self.pop:
            tensor_net = individual[1]
            size = len(tensor_net.get_middle_nodes())
            sum_sizes += size
        avg = sum_sizes / len(self.pop)
        return avg
