"""MNIST example using tensorflow dataset"""
import tensorflow as tf
from tensorEvolution import tensor_evolution


def main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    worker = tensor_evolution.EvolutionWorker.load('../pop.txt')  # change path as needed

    best = worker.get_best_individual()
    tensor_net = best[1]
    tensor_net.draw_graphviz_svg()
    model = tensor_net.build_model()
    model.summary()
    model.compile(loss=worker.master_config.loss, optimizer=worker.master_config.opt,
                  metrics=worker.master_config.config['metrics'])

    model.fit(x_train, y_train, epochs=10)
    model.evaluate(x_test, y_test)


if __name__ == "__main__":
    main()
