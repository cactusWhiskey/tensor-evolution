"""Module contains selection operators for use by evolution workers"""
import random


def _fitness_tournament(individuals: list, num_to_select: int,
                        t_size: int, random_selection=True) -> list:
    selected_individuals = []
    for _ in range(num_to_select):
        if random_selection:
            tournament_participants = random.choices(individuals, k=t_size)
        else:
            tournament_participants = individuals
        fittest = max(tournament_participants, key=lambda ind: ind.fitness)
        selected_individuals.append(fittest)
    return selected_individuals


def _complexity_tournament(individuals: list, num_to_select: int,
                           prob_sel_smallest: float,
                           random_selection=True, t_size=2) -> list:
    selected_individuals = []
    for _ in range(num_to_select):
        if random_selection:
            tournament_participants = random.choices(individuals, k=t_size)
        else:
            tournament_participants = individuals

        if random.random() < prob_sel_smallest:
            best = min(tournament_participants, key=lambda ind: ind[1].complexity)
        else:
            best = max(tournament_participants, key=lambda ind: ind[1].complexity)

        selected_individuals.append(best)
    return selected_individuals


def double_tournament(individuals: list, num_to_select: int,
                      fitness_t_size: int, prob_sel_smallest: float,
                      do_fitness_first: bool, complexity_t_size=2) -> list:
    """Runs a double tournament (complexity and fitness)
    Args:
        individuals: population to run the selection on
        num_to_select: total number of selections to make, typically equal to population size
        fitness_t_size: number of individuals that participate in a fitness tournament
        prob_sel_smallest: probability that the least complex individual gets
        selected in a complexity tournament
        do_fitness_first: do the fitness tournament first, then the complexity
        complexity_t_size: number of individuals in each complexity tournament

        :return A population of individuals
        """
    overall_winners = []
    if do_fitness_first:
        for _ in range(num_to_select):
            fitness_winners = _fitness_tournament(individuals, complexity_t_size, fitness_t_size)
            complexity_winner = _complexity_tournament(fitness_winners, 1, prob_sel_smallest,
                                                       random_selection=False)[0]
            overall_winners.append(complexity_winner)
    else:
        for _ in range(num_to_select):
            complexity_winners = _complexity_tournament(individuals,
                                                        fitness_t_size, prob_sel_smallest)
            fitness_winner = _fitness_tournament(complexity_winners, 1,
                                                 fitness_t_size, random_selection=False)[0]
            overall_winners.append(fitness_winner)

    return overall_winners
