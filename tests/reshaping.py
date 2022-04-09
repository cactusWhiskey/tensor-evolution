import tensorflow as tf
from tensorEvolution.nodes import tensor_node, node_utils

inputs = tf.keras.Input((8, 9))
flat = tf.keras.layers.Flatten()(inputs)

print("\n Reshaping test 1")
reshaped = node_utils.reshape_input(flat, 4)
print(f"Reshaped to: {reshaped.shape}")
print(f"Original shape: {inputs.shape}")

print("\n Reshaping test 2")
reshaped = node_utils.reshape_input(flat, 5)
print(f"Reshaped to: {reshaped.shape}")
print(f"Original shape: {inputs.shape}")

inputs = tf.keras.Input((4, 4, 4))
flat = tf.keras.layers.Flatten()(inputs)
print("\n Reshaping test 3")
reshaped = node_utils.reshape_input(flat, 5)
print(f"Reshaped to: {reshaped.shape}")
print(f"Original shape: {inputs.shape}")
