# Derived from Tensorflow basic regression example. The original work contained the following copyright/notice info:

# Copyright 2018 The TensorFlow Authors.

# @title Licensed under the Apache License, Version 2.0 (the "License");
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

# @title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# This derivative work is also licensed under Apache 2.0.

# Please see the original work for a detailed description of the steps not pertaining to evolution:
# https://www.tensorflow.org/tutorials/keras/regression

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorEvolution import evo_config, tensor_evolution


def main():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(url, names=column_names,
                              na_values='?', comment='\t',
                              sep=' ', skipinitialspace=True)
    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')

    train_features = train_features.to_numpy()
    train_labels = train_labels.to_numpy()
    test_features = test_features.to_numpy()
    test_labels = test_labels.to_numpy()

    del dataset, raw_dataset, train_dataset, test_dataset

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))
    train_features = normalizer(train_features).numpy()
    test_features = normalizer(test_features).numpy()

    #########################################################
    # end of tensorflow tutorial, beginning of evolution example

    # build custom config
    custom_config = {}
    custom_config['input_shapes'] = [[9]]
    custom_config['num_outputs'] = [1]
    custom_config['pop_size'] = 100
    custom_config['remote_mode'] = 'ray_remote'
    custom_config['loss'] = 'MeanAbsoluteError'
    custom_config['max_fit_epochs'] = 20
    custom_config['metrics'] = ['mean_absolute_error']
    custom_config['ngen'] = 20
    custom_config['direction'] = 'min'

    # set the custom config
    evo_config.master_config.setup_user_config(custom_config)

    # build data array
    data = np.array([train_features, train_labels, test_features, test_labels], dtype=object)
    # create evolution worker
    worker = tensor_evolution.EvolutionWorker()
    # run the evolution
    worker.evolve(data=data)

    # get best individual from the population
    best = worker.get_best_individual()
    tensor_net = best[1]
    # draw the genome
    tensor_net.draw_graphviz_svg()
    # build a tf model from the genome
    model = tensor_net.build_model()

    model.compile(loss=worker.master_config.loss, optimizer=worker.master_config.opt,
                  metrics=worker.master_config.config['metrics'])
    print(f"Validation metrics without additional training (using saved weights) "
          f"{model.evaluate(test_features, test_labels)}")

    model.fit(train_features, train_labels, epochs=50)
    print(f"Validation metrics after additional training "
          f"{model.evaluate(test_features, test_labels)}")
    worker.plot()


if __name__ == "__main__":
    main()
