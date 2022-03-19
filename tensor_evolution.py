import random
import numpy
import ray
from deap import creator, base, tools
from matplotlib import pyplot as plt
from ActorPoolExtension import ActorPoolExtension
import config_utils
import tensor_network
import tensor_node


def setup_creator():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)


@ray.remote(num_cpus=1)
class RemoteEvoActor:
    def __init__(self, data, remote_config: dict):
        self.data = data
        self.remote_config = remote_config

    def eval(self, individual: list):
        x_train, y_train, x_test, y_test = self.data
        tn = individual[1]
        model = tn.build_model()
        model.compile(loss=self.remote_config['loss'],
                      optimizer=self.remote_config['opt'],
                      metrics=self.remote_config['metrics'])
        model.fit(x_train, y_train, epochs=self.remote_config['epochs'],
                  callbacks=self.remote_config['callbacks'])
        test_loss, test_acc = model.evaluate(x_test, y_test)

        length = len(individual[1].get_middle_nodes())
        complexity_penalty = self.remote_config['complexity_penalty']
        penalty = complexity_penalty * length
        return (test_acc - penalty),


# def cx_chain(ind1, ind2):
#     tn = ind1[1]
#     other_tn = ind2[1]
#     tensor_network.cx_chain(tn, other_tn)
#     return ind1, ind2


class EvolutionWorker:
    def __init__(self):
        self.record = None
        self.logbook = None
        self.stats = None
        self.pop = None
        self.pool = None
        setup_creator()
        self.toolbox = base.Toolbox()
        self.setup_toolbox()
        self.setup_stats()
        self.setup_log()

    def get_remote_config(self) -> dict:
        remote_config = {}
        remote_config['loss'] = config_utils.loss
        remote_config['metrics'] = config_utils.config['metrics']
        remote_config['opt'] = config_utils.opt
        remote_config['epochs'] = config_utils.config['max_fit_epochs']
        remote_config['callbacks'] = config_utils.callbacks
        remote_config['complexity_penalty'] = config_utils.config['complexity_penalty']
        return remote_config

    def setup_toolbox(self):
        self.toolbox.register("individual", self.initialize_ind,
                              input_shapes=config_utils.config['input_shapes'],
                              num_outputs=config_utils.config['num_outputs'])
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", self.cx_single_node)
        self.toolbox.register("mutate_insert", self.mutate_insert)
        self.toolbox.register("mutate_delete", self.mutate_delete)
        self.toolbox.register("mutate_mutate", self.mutate_mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=config_utils.config['t_size'])

    def setup_stats(self):
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        self.stats.register("avg", numpy.mean)
        # stats.register("std", numpy.std)
        self.stats.register("min", numpy.min)
        self.stats.register("max", numpy.max)

    def setup_log(self):
        self.logbook = tools.Logbook()
        self.logbook.header = "gen", "avg", "max", "min"

    def plot(self):
        gen, avg = self.logbook.select("gen"), self.logbook.select("avg")
        fig, ax = plt.subplots()
        ax.plot(gen, avg, "g-", label="Avg Fitness")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness", color="b")
        plt.show()

    @staticmethod
    def initialize_ind(input_shapes, num_outputs) -> list:
        ind = creator.Individual()
        hyper_params = []
        tn = tensor_network.TensorNetwork(input_shapes, num_outputs)
        ind.append(hyper_params)
        ind.append(tn)
        return ind

    @staticmethod
    def eval_remote(actor: RemoteEvoActor, individual: list):
        individual = list(individual)
        return actor.eval.remote(individual)

    @staticmethod
    def evaluate(individual: list, data: tuple):
        x_train, y_train, x_test, y_test = data
        tn = individual[1]

        model = tn.build_model()
        model.compile(loss=config_utils.loss, optimizer=config_utils.opt,
                      metrics=config_utils.config['metrics'])

        if config_utils.config['global_cache_training']:
            tn.store_weights(model, direction_into_tn=False)

        model.fit(x_train, y_train, epochs=config_utils.config['max_fit_epochs'],
                  callbacks=config_utils.callbacks)
        test_loss, test_acc = model.evaluate(x_test, y_test)

        if config_utils.config['global_cache_training']:
            tn.store_weights(model, direction_into_tn=True)

        length = len(individual[1].get_middle_nodes())
        penalty = config_utils.config['complexity_penalty'] * length

        return test_acc - penalty,

    @staticmethod
    def mutate_insert(individual: list):
        tn = individual[1]
        position = random.randint(0, len(tn.get_valid_insert_positions()) - 1)
        node_type = random.choice(config_utils.config['valid_node_types'])
        node = tensor_node.create(node_type)
        tn.insert_node(node, position)
        return individual,

    @staticmethod
    def mutate_mutate(individual: list):
        tn = individual[1]
        length = len(tn.get_mutatable_nodes())

        if length == 0:  # nothing to mutate
            return individual,

        position = random.randint(0, length - 1)
        tn.mutate_node(position)
        return individual,

    @staticmethod
    def mutate_delete(individual: list):
        tn = individual[1]
        length = len(tn.get_middle_nodes())

        if length == 0:  # nothing to delete
            return individual,

        position = random.randint(0, length - 1)
        node_id = list(tn.get_middle_nodes().keys())[position]
        tn.delete_node(node_id=node_id)
        return individual,

    @staticmethod
    def cx_single_node(ind1, ind2):
        tn = ind1[1]
        other_tn = ind2[1]
        tensor_network.cx_single_node(tn, other_tn)
        return ind1, ind2

    def evolve(self, data):
        if not config_utils.config['remote']:
            self.toolbox.register("evaluate", self.evaluate, data=data)
        else:
            ray.init()
            actors = []
            for _ in range(config_utils.config['remote_actors']):
                actor = RemoteEvoActor.remote(data, self.get_remote_config())
                actors.append(actor)

            self.pool = ActorPoolExtension(actors)
            self.toolbox.register("evaluate", self.eval_remote)

        self._evolve()

    def _evolve(self):
        self.pop = self.toolbox.population(n=config_utils.config['pop_size'])

        # Evaluate the entire population
        if config_utils.config['remote']:
            fitnesses = self.pool.map_ordered_return_all(self.toolbox.evaluate, self.pop)
        else:
            fitnesses = list(map(self.toolbox.evaluate, self.pop))

        for ind, fit in zip(self.pop, fitnesses):
            ind.fitness.values = fit

        for gen in range(config_utils.config['ngen']):
            # Select the next generation individuals
            offspring = self.toolbox.select(self.pop, len(self.pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < config_utils.config['cx']:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # apply mutation
            for mutant in offspring:
                if random.random() < config_utils.config['m_insert']:
                    self.toolbox.mutate_insert(mutant)
                    del mutant.fitness.values

                if random.random() < config_utils.config['m_del']:
                    self.toolbox.mutate_delete(mutant)
                    del mutant.fitness.values

                if random.random() < config_utils.config['m_mut']:
                    self.toolbox.mutate_mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            if config_utils.config['remote']:
                fitnesses = self.pool.map_ordered_return_all(self.toolbox.evaluate, invalid_ind)
            else:
                fitnesses = list(map(self.toolbox.evaluate, invalid_ind))

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            # The population is entirely replaced by the offspring
            self.pop[:] = offspring
            self.record = self.stats.compile(self.pop)
            self.logbook.record(gen=gen, **self.record)
            print('\n')
            print(self.logbook.header)
            print(self.logbook.stream)
            print('\n')

        print("-- End of (successful) evolution --")

        best_ind = tools.selBest(self.pop, 1)[0]
        print("Best individual is %s" % best_ind.fitness.values)
        best_ind[1].build_model().summary()
        #self.plot()
