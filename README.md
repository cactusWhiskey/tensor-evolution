# Tensor Evolution


Tensor-Evolution is a library for evolving neural network topology using a genetic algorithm. This library currently 
uses [Deap](https://github.com/DEAP/deap) as its evolutionary backend, and [Tensorflow](https://github.com/tensorflow/tensorflow) 
for the neural networks.<br>

Note that this library doesn't build networks a single neuron at a time, the basic building blocks are entire layers.


## Philosophy

Population members start as the input layer connected directly to the output layer. Mutation operators exist for 
inserting layers (from a list of supported types), deleting layers, and for mutating an existing layer's properties. A 
crossover operator is also implemented.

Fitness is evaluated by building, compiling, and training a model from each population member's genome. 
Training is done the standard way (i.e. via backpropagation, not through any evolutionary means).

Note that most layer types can be added amost anywhere in the genome. If the input shape isn't right, it's corrected 
(attempts are made to correct it intelligently, but if required it's forced to fit). 

## Supported Layer Types



This list is currently expanding. So far:

- Dense
- ReLu
- Conv2D, 3D
- Maxpool2D, 3D
- Addition
- BatchNorm
- Flatten
- LSTM
- GlobalAvgPooling 1D
- Embedding
- Concat

## Installation


```pip install tensor-evolution ```

## Usage



### Running an Evolution
Start by importing the *tensor_evolution* module. This is the main driver for the evolution. 

```import tensor_evolution```

Next, prepare your data as a tuple of four objects, like so:

```data = x_train, y_train, x_test, y_test```

Then create an evolution worker, and use that worker to drive the evolution:

```worker = tensor_evolution.EvolutionWorker()```
```worker.evolve(data=data)```

Please reference the end to end examples for full details.

### Configuration

Everything is configured via yaml file. For the moment, since you will need to clone the project to use it, 
just edit the default *config.yaml* file.

For example, to change population size to 30:

```
####
# Evolution Controls
####
...
pop_size: 30 #population size

```

Mutation rates, valid neural network layer types, **input and output shapes**, etc. are all controlled from the config file.

## Project Status


Very much still a work in progress, (as is this readme), but it is functional. The mnist example runs just fine.

## Dependencies

| Library                                                | License                                                                                        |
|--------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [tensorflow](https://github.com/tensorflow/tensorflow) | [Apache License 2.0](https://github.com/tensorflow/tensorflow/blob/master/LICENSE)             |
| [networkx](https://github.com/networkx/networkx)       | [BSD 3-Clause](https://github.com/networkx/networkx/blob/main/LICENSE.txt)                     |
| [ray](https://github.com/ray-project/ray)              | [Apache License 2.0](https://github.com/ray-project/ray/blob/master/LICENSE)                   |
| [numpy](https://github.com/numpy/numpy)                | [BSD 3-Clause](https://github.com/numpy/numpy/blob/main/LICENSE.txt)                           |
| [deap](https://github.com/DEAP/deap)                   | [GNU Lesser General Public License v3.0](https://github.com/DEAP/deap/blob/master/LICENSE.txt) |
| [matplotlib](https://github.com/matplotlib/matplotlib) | [License Details](https://matplotlib.org/3.5.0/users/project/license.html#license-agreement)   |
| [sympy](https://github.com/sympy/sympy)                | [License Details](https://github.com/sympy/sympy/blob/master/LICENSE)                          |
| [graphviz](https://github.com/graphp/graphviz)         | [MIT License](https://github.com/graphp/graphviz/blob/master/LICENSE)                          |


## MNIST Results

The best individual after running MNIST with a population of 20 individuals for 15 generations:

![MNIST Genome](/doc/images/MNIST.svg) 

![MNIST Summary](/doc/images/MNIST_Summary.png)