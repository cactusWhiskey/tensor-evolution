"""Utilities for evolution"""
import tensorflow as tf


def compute_complexity(model: tf.keras.Model) -> int:
    """Returns the total trainable variables for the model"""
    total_trainable = 0
    flat_shape = 1
    for variable in model.trainable_variables:
        for dim in variable.shape:
            flat_shape *= dim
        total_trainable += flat_shape
        flat_shape = 1
    return total_trainable
