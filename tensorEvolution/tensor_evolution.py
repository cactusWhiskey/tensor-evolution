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
        self.initial_nodes = None
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
        # self.toolbox.register("select", tools.selTournament,
        #                       tournsize=7)
        self.toolbox.register("select", selection.double_tournament,
                              fitness_t_size=self.master_config.config['fitness_t_size'],
                              prob_sel_smallest=self.master_config.config['prob_sel_least_complex'],
                              do_fitness_first=True,
                              complexity_t_size=self.master_config.config['complexity_t_size'])

    def _setup_stats(self):
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", numpy.mean)
        # stats.register("std", numpy.std)
        self.stats.register("min", numpy.min)
        self.stats.register("max", numpy.max)

    def _setup_log(self):
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "avg", "max", "min", "max_size", "avg_size"

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
                                                  preprocessing_layers=self.preprocessing_layers,
                                                  initial_nodes=self.initial_nodes)
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
        if self.master_config.config['remote_mode'] == "local":
            self.toolbox.register("evaluate", evaluation.evaluate, data=data)
        else:
            ray.init()
            # actors = []
            # for _ in range(self.master_config.config['remote_actors']):
            #     actor = RemoteEvoActor.remote(data)
            #     actors.append(actor)
            # self.pool = ActorPool(actors)

            # put the data into the ray object store
            data_id = ray.put(data)
            # self.toolbox.register("evaluate", evaluation.eval_remote)
            self.toolbox.register("evaluate", evaluation.eval_ray_task.remote, data_id=data_id)

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

        # create a tuple the contains each individual paired with its index
        # this will enable us to retrieve the evaluation results in
        # any order, and still know which fitness goes with
        # which individual
        # also convert each individual to a list for serialization
        indexes_with_inds = [(index, list(ind)) for index, ind in enumerate(individuals)]

        # check if remote
        remote_mode = self.master_config.config['remote_mode']

        if remote_mode == "ray_remote":
            # use remote function
            # eval_results = list(self.pool.map_unordered(self.toolbox.evaluate, indexes_with_inds))

            # this is a list of reference ids, not the actual results
            eval_results_ids = list(map(self.toolbox.evaluate, indexes_with_inds))
            # need to get a result that is done
            while len(eval_results_ids):
                # get a done result id
                done_id, eval_results_ids = ray.wait(eval_results_ids)
                # process the result
                result = ray.get(done_id[0])
                self._process_evaluation_result(result, remote_mode, individuals)
        else:
            # use local function
            eval_results = list(map(self.toolbox.evaluate, indexes_with_inds))
            # local evaluation
            for result in eval_results:
                self._process_evaluation_result(result, remote_mode, individuals)

    @staticmethod
    def _process_evaluation_result(result: tuple, remote_mode: str, individuals):
        # fitness tuple is always in the zero position of the evaluation returned results
        fitness = result[0]
        # index is in the 2 position
        index = result[2]
        # get the individual this fitness result is for
        this_individual = individuals[index]
        # set the fitness
        this_individual.fitness.values = fitness

        if remote_mode == "ray_remote":
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
        biggest, avg = self._get_individual_size_stats()
        self.record['max_size'] = biggest
        self.record['avg_size'] = avg
        self.logbook.record(gen=gen, **self.record)
        print(self.logbook.stream)

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
        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        self._evaluation_on_individuals(invalid_ind)

    def _main_evolution_loop(self):
        for gen in range(self.master_config.config['ngen']):
            self._pop_integrity_check()
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
        new_worker.pop = EvolutionWorker.deserialize_pop(serial_dict['pop'])
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
    def deserialize_individual(serial_individual: list):
        """Rebuilds an individual from serial form"""
        # pylint: disable=E1101
        new_ind = creator.Individual()
        individual_config = evo_config.EvoConfig.deserialize(serial_individual[0])
        tensor_net = tensor_network.TensorNetwork.deserialize(serial_individual[1])
        new_ind.append(individual_config)
        new_ind.append(tensor_net)
        new_ind.fitness.values = tuple(serial_individual[2])
        return new_ind

    @staticmethod
    def deserialize_pop(serial_pop: list) -> list:
        """Rebuilds population from serial form"""
        new_pop = []
        for serial_individual in serial_pop:
            new_pop.append(EvolutionWorker.deserialize_individual(serial_individual))
        return new_pop

    def setup_preprocessing(self, preprocessing_layers: list):
        """Use this method to set preprocessing layers on the input
        Args:
            preprocessing_layers: needs to be a list of lists of layers.
            The layers need to be ready to go (so run any adapting beforehand)"""
        self.preprocessing_layers = preprocessing_layers

    def _save_preprocessing_layers(self):
        if self.preprocessing_layers is None:
            return

        for index, layer_stack in enumerate(self.preprocessing_layers):
            model = tf.keras.Sequential()
            for pre_layer in layer_stack:
                model.add(pre_layer)
            preprocessing_save_path = evo_config. \
                master_config.config['preprocessing_save_path'][index]
            model.save(preprocessing_save_path)

    # def _load_preprocessing_layers(self):
    #     self.preprocessing_layers = []
    #     preprocessing_save_paths = evo_config.master_config.config['preprocessing_save_path']
    #     for path in preprocessing_save_paths:
    #         model = tf.keras.models.load_model(path)
    #         prelayers = []
    #         for layer in model.layers:
    #             prelayers.append(layer)
    #         self.preprocessing_layers.append(prelayers)

    def _get_individual_size_stats(self) -> tuple:
        """For debugging use, returns info on individual size.
        Useful for checking if bloat control is working."""
        biggest = 0
        sum_sizes = 0
        for individual in self.pop:
            tensor_net = individual[1]
            size = len(tensor_net.get_middle_nodes())
            biggest = max(biggest, size)
            sum_sizes += size
        avg = sum_sizes / len(self.pop)
        return biggest, avg

    def set_initial_nodes(self, nodes: list):
        """Add initial nodes to all networks
        :param nodes: list of lists of initial nodes
        """
        self.initial_nodes = nodes

    def _pop_integrity_check(self):
        """Debug hook to check on the population"""
        pass
