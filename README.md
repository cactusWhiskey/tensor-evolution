# Tensor Evolution
Tensor-Evolution is a library for evolving neural network topology using a genetic algorithm. This library currently 
uses [Deap](https://github.com/DEAP/deap) as its evolutionary backend, and [Tensorflow](https://github.com/tensorflow/tensorflow) 
for the neural networks. 

## Installation
At the moment you'll need to clone the source and then either edit one of the examples, or create a new python file 
and import the *tensor_evolution* module. I am working on getting this project on pip. 

## Usage

### Running an Evolution
Start by importing the *tensor_evolution* module. This is the main driver for the evolution. 

```
import tensor_evolution
```

Next, prepare your data as a tuple of four objects, like so:

```
data = x_train, y_train, x_test, y_test
```

Then create an evolution worker, and use that worker to drive the evolution:
```
worker = tensor_evolution.EvolutionWorker()
worker.evolve(data=data)
```

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

## License 