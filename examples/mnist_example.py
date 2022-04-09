"""MNIST example using tensorflow dataset"""
import tensorflow as tf
from tensorEvolution import tensor_evolution
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

data = x_train, y_train, x_test, y_test
worker = tensor_evolution.EvolutionWorker()

worker.evolve(data=data)
