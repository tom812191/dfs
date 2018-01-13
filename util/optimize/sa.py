import numpy as np
from functools import partial

from util.optimize.evaluate import evaluate_lineup_set


def optimize_lineup_set(lineup_universe, simulated_scores, num_lineups=100, verbose=0, store_history=False, target=330,
                        initial_state=None, initial_temperature=10, end_temperature=0.00001, cooling_rate=0.003):

    if initial_state is None:
        initial_state = get_initial_state(lineup_universe, num_lineups)

    eval = partial(evaluate_lineup_set, lineup_universe, simulated_scores, target=target, as_tuple=False)

    history = []

    temperature = initial_temperature
    current_state = initial_state
    current_fitness = eval(current_state)

    best_state = current_state
    best_fitness = current_fitness

    while temperature > end_temperature:
        if verbose > 0:
            print('Temp: {} of {}'.format(temperature, end_temperature))
            print('Fitness: {}  (best: {})'.format(current_fitness, best_fitness))

        new_state = get_new_state(current_state, lineup_universe)
        new_fitness = eval(new_state)

        acceptance_prob = acceptance_probability(current_fitness, new_fitness, temperature)
        if np.random.rand() < acceptance_prob:
            if verbose > 1:
                if new_fitness < current_fitness:
                    print('Accepting worse with prob {}'.format(acceptance_prob))

            current_state = new_state
            current_fitness = new_fitness

            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_state = current_state

        temperature *= 1 - cooling_rate

        if store_history:
            history.append([temperature, current_fitness, best_fitness])

    return lineup_universe[best_state], best_state, history


def acceptance_probability(current_fitness, new_fitness, temperature):
    if new_fitness > current_fitness:
        return 1.0

    return np.exp((new_fitness - current_fitness) / temperature)


def get_initial_state(lineup_universe, num_lineups):
    return np.random.randint(len(lineup_universe), size=num_lineups)


def get_new_state(current_state, lineup_universe):
    state_idx = np.random.randint(len(current_state))
    new_lineup = np.random.randint(len(lineup_universe))

    new_state = np.copy(current_state)
    new_state[state_idx] = new_lineup

    return new_state
