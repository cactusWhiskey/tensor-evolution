"""Simple test to verify tensor_evolution module imports and a worker can be initialized"""

from tensorEvolution import tensor_evolution


def main():
    # create evolution worker
    worker = tensor_evolution.EvolutionWorker()


if __name__ == "__main__":
    main()
