"""Simple test to verify tensor_evolution module imports and a worker can be initialized"""

from tensorEvolution import tensor_evolution


def test_basic_setup():
    worker = tensor_evolution.EvolutionWorker()
    assert True
