import random
from deap import creator, base, tools

from functools import partial

from util.optimize.evaluate import evaluate_lineup_set


def optimize_lineup_set(lineup_universe, simulated_scores, num_lineups=100, target=330,
                        ga_pop_size=1000, ga_num_gen=100, ga_prob_cross=0.5, ga_prob_mut=0.2):
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # An individual is a list of indexes within the lineup_universe
    toolbox.register('attr_lineups', random.sample, range(len(lineup_universe)), num_lineups)
    toolbox.register('individual', tools.initIterate, creator.Individual,
                     toolbox.attr_lineups)

    # The population is a list of individuals
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # Evaluation function
    eval = partial(evaluate_lineup_set, lineup_universe, simulated_scores, target=target)
    toolbox.register('evaluate', eval)

    # Crossover
    toolbox.register('mate', tools.cxTwoPoint)

    # Mutation
    toolbox.register('mutate', tools.mutUniformInt, low=0, up=(len(lineup_universe) - 1), indpb=0.05)

    # Selection
    toolbox.register("select", tools.selTournament, tournsize=5)

    # Initialize Population
    pop = toolbox.population(n=ga_pop_size)

    # Begin evolution
    print('Begin Evolution')
    print('Target Score: {}'.format(target))
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(ga_num_gen):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability ga_prob_cross
            if random.random() < ga_prob_cross:
                toolbox.mate(child1, child2)

                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability ga_prob_mut
            if random.random() < ga_prob_mut:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        print("  Evaluated %i individuals" % len(invalid_ind))

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        print('Generation {} of {}'.format(g, ga_num_gen))

    print("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    return lineup_universe[list(best_ind)], list(best_ind)
