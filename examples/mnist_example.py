"""MNIST example using tensorflow dataset.

Derived from Tensorflow quickstart beginner example. The original work contained the following copyright/notice info:

Copyright 2019 The TensorFlow Authors.
#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

This derivative work is also licensed under Apache 2.0.
"""
import tensorflow as tf
from tensorEvolution import tensor_evolution


def main():
    # get mnist dataset
    mnist = tf.keras.datasets.mnist

    # scale the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # pack into data tuple
    data = x_train, y_train, x_test, y_test

    # create evolution worker
    worker = tensor_evolution.EvolutionWorker()
    # evolve
    worker.evolve(data=data)


if __name__ == "__main__":
    main()
