"""Text classification example"""
# Derived from Tensorflow basic text classification example.
# The original work contained the following
# copyright/notice info:

# MIT License
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
# https://www.tensorflow.org/tutorials/keras/text_classification

import os
import re
import string

import tensorflow as tf
import tensorflow_datasets as tfds

from configs.__init__ import CONFIG_DIR
from tensorEvolution import evo_config, tensor_evolution
from tensorEvolution.nodes import embedding


def main():
    """Main method"""

    # load data
    train_data, test_data = tfds.load(name="imdb_reviews", split=["train", "test"],
                                      batch_size=-1, as_supervised=True)

    # convert to numpy (evolution isn't supporting tf datasets at the moment)
    train_examples, train_labels = tfds.as_numpy(train_data)
    test_examples, test_labels = tfds.as_numpy(test_data)

    # see this example on the tensorflow tutorials for an explanation
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation),
                                        '')

    max_features = 10000
    sequence_length = 250

    vectorize_layer = tf.keras.layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    vectorize_layer.adapt(train_examples)

    def vectorize_text(text):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text)

    # vectorize the text
    train_examples = vectorize_text(train_examples)
    test_examples = vectorize_text(test_examples)

    embedding_dim = 16

    # load the evolution config
    path = os.path.join(CONFIG_DIR, 'text_classification_config.yaml')
    evo_config.master_config.setup_user_config(path)

    # crate data tuple
    data = train_examples, train_labels, test_examples, test_labels
    # create evolution worker
    worker = tensor_evolution.EvolutionWorker()

    # build an embedding node as an initial node
    embedding_node = embedding.EmbeddingNode(max_features + 1, embedding_dim)
    initial_node_stack = [embedding_node]
    # provide it as a list of lists
    worker.set_initial_nodes([initial_node_stack])
    # run the evolution
    worker.evolve(data=data)

    best = worker.get_best_individual()
    print("\n" + str(best[0]))
    # tensor_net = best[1]
    # model = tensor_net.build_model()
    # model.compile(loss=worker.master_config.loss, optimizer=worker.master_config.opt,
    #               metrics=worker.master_config.config['metrics'])
    #
    # model.fit(train_features, train_labels, epochs=20)
    # model.evaluate(test_features, test_labels)


if __name__ == "__main__":
    main()
