"""This module contains the main evolution loop"""
import json
import os
import random
import sys
import time

import numpy as np
import ray
import tensorflow as tf
from deap import creator, base, tools
from matplotlib import pyplot as plt

from tensorEvolution import evo_config
from tensorEvolution import tensor_encoder
from tensorEvolution import tensor_network
from tensorEvolution.evolutionOperators import selection, crossover, mutation, evaluation

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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
        self.data_for_gen_batching = None
        self.data = None
        self.num_to_skip = 0
        self.val_num_to_skip = 0
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
        self.stats.register("avg", np.mean)
        # stats.register("std", numpy.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

    def _setup_log(self):
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "avg", "max", "min", "max_size", "avg_size", 'gen_time'

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

    def evolve(self, data, custom_init=False):
        """Main evolution method. Call to begin evolution.

        :param data: data which will be passed to the model.fit method for training.
                Can be either data or a reference to a ray object store object
        :param custom_init: set to True if you've already initialized ray manually.
                Data should be a reference to an object store object in this case
        """
        # pylint: disable=E1101

        batch_data_per_gen = self.master_config.config['batch_data_per_gen']
        if batch_data_per_gen:
            # move the data over so that the self.data variable can be used for the batches
            self.data_for_gen_batching = data
            data = None

        # configure the evaluation function based on remote vs local execution
        if self.master_config.config['remote_mode'] == "local":
            self.toolbox.register("evaluate", evaluation.evaluate)
        else:
            # remote mode
            if not custom_init:
                ray.init()

                # put the data into the ray object store
                data = ray.put(data)
                # self.toolbox.register("evaluate", evaluation.eval_remote)

            self.toolbox.register("evaluate", evaluation.eval_ray_task.remote)

        self.data = data
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

        # check if remote
        remote_mode = self.master_config.config['remote_mode']
        batch_data_per_gen = self.master_config.config['batch_data_per_gen']

        # pull a new batch of data
        if batch_data_per_gen:
            train_num_to_take = self.master_config.config['batch_data_train_size']
            validation_num_to_take = self.master_config.config['batch_data_val_size']
            data_type = self.master_config.config['gen_batch_datatype']
            if data_type == 'tf_data':
                if remote_mode == "ray_remote":
                    raise ValueError("tf data not compatible with ray serialization, please use ray_data")

                tf_dataset, val_tf_dataset = self.data_for_gen_batching
                current_dataset = tf_dataset.skip(self.num_to_skip)
                train_dataset = current_dataset.take(train_num_to_take).batch(32)
                val_current_dataset = val_tf_dataset.skip(self.val_num_to_skip)
                validation_dataset = val_current_dataset.take(validation_num_to_take).batch(32)
                self.num_to_skip += train_num_to_take
                self.val_num_to_skip += validation_num_to_take

                self.data = np.array([train_dataset, validation_dataset])

            elif data_type == 'numpy_data':
                train_features = self.data_for_gen_batching[0]
                train_labels = self.data_for_gen_batching[1]
                val_features = self.data_for_gen_batching[2]
                val_labels = self.data_for_gen_batching[3]

                train_features, train_labels, self.num_to_skip = self._generate_numpy_gen_batch(train_features,
                                                                                                train_labels,
                                                                                                train_num_to_take,
                                                                                                self.num_to_skip)
                val_features, val_labels, self.val_num_to_skip = self._generate_numpy_gen_batch(val_features,
                                                                                                val_labels,
                                                                                                validation_num_to_take,
                                                                                                self.val_num_to_skip)

                data = (train_features, train_labels, val_features, val_labels)

                if remote_mode == "ray_remote":
                    self.data = ray.put(data)
                else:
                    self.data = data

            else:
                raise ValueError("Unsupported gen batch data type")

        if remote_mode == "ray_remote":
            # use remote function
            # eval_results = list(self.pool.map_unordered(self.toolbox.evaluate, indexes_with_inds))

            # these are lists of reference ids, not the actual results
            submitted_ids = []
            results_ids = []
            for index, individual in enumerate(individuals):
                # this loop submits all the tasks to ray

                # convert to list for serialization
                individual = list(individual)

                num_in_flight = len(submitted_ids)
                max_allowed_tasks = self.master_config.config['max_ray_tasks']

                # check if we have submitted too many tasks already
                if num_in_flight > max_allowed_tasks:
                    # too many tasks in flight, wait (could crash the program otherwise
                    # with 'out of memory' error
                    num_to_wait_for = num_in_flight - max_allowed_tasks
                    # this will block until num_to_wait_for results finish evaluation
                    done_ids, submitted_ids = ray.wait(submitted_ids, num_returns=num_to_wait_for)
                    # add the done ids to the results_ids list
                    results_ids += done_ids
                # if we get this far, we don't have too many tasks in flight, so submit another one
                indexed_individual = (index, individual)
                submitted_ids.append(self.toolbox.evaluate(indexed_individual, self.data))

            # at this point all tasks have been submitted, and many should already be finished
            # lets start processing results while the remaining evaluations finish
            for result_id in results_ids:
                # get the actual evaluation result
                result = ray.get(result_id)
                self._process_evaluation_result(result, remote_mode, individuals)

            # out of results to process, wait for more
            # need to get a result that is done
            while len(submitted_ids):
                # get a done result id
                done_id, submitted_ids = ray.wait(submitted_ids)
                # process the result
                result = ray.get(done_id[0])
                self._process_evaluation_result(result, remote_mode, individuals)
        else:
            # use local function
            eval_results = []
            for index, individual in enumerate(individuals):
                indexed_individual = (index, list(individual))
                eval_results.append(self.toolbox.evaluate(indexed_individual, self.data))
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
            # so complexity and weight cache, and integrity check didn't get done on the local (master) copy of
            # the individuals. Need to do that now.

            # __dict__ from the remote individual is returned by the eval function at position 1
            tensor_dict_from_remote = result[1]
            tensor_net = this_individual[1]  # our local copy of this individual
            # copy the updated state over
            tensor_net.__dict__ = tensor_dict_from_remote

    def _gen_bookkeeping(self, gen, gen_time):
        self.record = self.stats.compile(self.pop)
        biggest, avg = self._get_individual_size_stats()
        self.record['max_size'] = biggest
        self.record['avg_size'] = avg
        gen_time = f"{round(gen_time)}s"
        self.record['gen_time'] = gen_time
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
            # self._save_preprocessing_layers()

        # compute initial fitness of population
        invalid_ind = [ind for ind in self.pop if not ind.fitness.valid]
        self._evaluation_on_individuals(invalid_ind)

    def _main_evolution_loop(self):
        for gen in range(self.master_config.config['ngen']):
            gen_start_time = time.time()
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

            gen_done_time = time.time()
            gen_time = gen_done_time - gen_start_time
            # Record and print info about this generation
            self._gen_bookkeeping(gen, gen_time)

            # save data to disk
            self._save_every(gen)

    def _post_evolution_tasks(self):
        print("-- End of (successful) evolution --")

        best_ind = tools.selBest(self.pop, 1)[0]
        print("Best individual is %s" % best_ind.fitness.values)
        best_ind[1].build_model().summary()
        self.save(self.master_config.config['save_pop_filepath'])

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

    @staticmethod
    def _generate_numpy_gen_batch(np_features: np.ndarray, np_labels: np.ndarray,
                                  num_to_take: int, num_to_skip: int) -> tuple:

        if len(np_features) != len(np_labels):
            raise ValueError("datasets must have same length")
        data_len = len(np_features)

        gen_features = None
        gen_labels = None
        done = False
        while not done:
            # deal with train dataset indexes
            start_index = num_to_skip % data_len
            end_index = start_index + num_to_take
            if end_index > data_len:
                if gen_features is None:
                    gen_features = np_features[start_index:]
                    gen_labels = np_labels[start_index:]
                else:
                    gen_features = np.concatenate((gen_features, np_features[start_index:]), axis=0)
                    gen_labels = np.concatenate((gen_labels, np_labels[start_index:]), axis=0)

                num_taken = data_len - start_index
                num_to_take -= num_taken
                num_to_skip += num_taken
            else:
                if gen_features is None:
                    gen_features = np_features[start_index:end_index]
                    gen_labels = np_labels[start_index:end_index]
                else:
                    gen_features = np.concatenate((gen_features, np_features[start_index:end_index]), axis=0)
                    gen_labels = np.concatenate((gen_labels, np_labels[start_index:end_index]), axis=0)
                num_taken = end_index - start_index
                num_to_take -= num_taken
                num_to_skip += num_taken
                done = True
        return gen_features, gen_labels, num_to_skip
