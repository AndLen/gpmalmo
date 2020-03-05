from math import ceil

from deap import base
from deap import creator
from deap.tools import ParetoFront
from sklearn.metrics import pairwise_distances

import gptools.weighted_generators as wg
from gpmalmo import rundata as rd
from gpmalmo.eval import evalGPMalNC
from gpmalmo.gp_design import get_pset_weights
from gpmalmo.gpmalnc_moead import GPMALNCMOEAD
from gptools.ParallelToolbox import ParallelToolbox
from gptools.gp_util import *
from gptools.multitree import *
from gptools.util import init_data, final_output


def main():
    pop = toolbox.population(n=rd.pop_size)
    stats_cost = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats_num_trees = tools.Statistics(lambda ind: ind.fitness.values[1])
    mstats = tools.MultiStatistics(cost=stats_cost, num_trees=stats_num_trees)
    mstats.register("min", np.min, axis=0)
    mstats.register("median", np.median, axis=0)
    mstats.register("max", np.max, axis=0)
    hof = ParetoFront()
    this_moead = GPMALNCMOEAD(rd.data_t, pop, toolbox, len(pop), rd.cxpb, rd.mutpb, rd,
                              ngen=rd.gens, stats=mstats,
                              halloffame=hof, verbose=True, adapative_mute_ERC=False)
    pop, logbook, hof = this_moead.execute()
    return pop, mstats, hof, logbook


def pick_nns(rd, step_length=10):
    i = 0
    indicies = []
    # this can probably just be a for loop if I derive the no. iterations instead of being lazy
    while True:
        base = step_length * ((2 ** i) - 1)
        step_multiplier = 2 ** i
        for j in range(step_length):
            next = base + (step_multiplier * j)
            #print(next)
            ##yeah yeah, it's easier...
            if next >= rd.num_instances:
                return indicies
            if next != 0:
                indicies.append(next)
        i+=1




def make_ind(toolbox, creator, max_trees):
    return creator.Individual([toolbox.tree() for _ in range(random.randint(1, max_trees))])
if __name__ == "__main__":
    init_data(rundata)
    max_trees = max(2,ceil(.5 * rd.num_features))
    rd.num_trees = max_trees#min(max_trees,20)
    pset, weights = get_pset_weights(rd.data, rd.num_features, rd)
    rd.pset = pset
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * rd.nobj)
    creator.create("Individual", list, fitness=creator.FitnessMin, pset=pset)

    toolbox = ParallelToolbox()  #


    toolbox.register("expr", wg.w_genHalfAndHalf, pset=pset, weighted_terms=weights, min_=0, max_=rd.max_depth)
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register("individual",make_ind,toolbox,creator,rd.num_trees)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evalGPMalNC, rd.data_t, toolbox)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("mate", lim_xmate_aic)

    toolbox.register("expr_mut", wg.w_genFull, weighted_terms=weights, min_=0, max_=rd.max_depth)
    toolbox.register("mutate", lim_xmut, expr=toolbox.expr_mut)

    toolbox.register("mutate_ar", mutate_add_remove, rd.num_trees, toolbox.tree)

    # GPMAL stuff.
    rd.pairwise_distances = pairwise_distances(rd.data)
    rd.ordered_neighbours = np.argsort(rd.pairwise_distances, axis=1)

    # get a list of indicies to use
    rd.neighbours = np.array(pick_nns(rd,step_length=10))#20))
    rd.identity_ordering = np.array([x for x in range(len(rd.neighbours))])
    rd.all_orderings = rd.ordered_neighbours[:,rd.neighbours]
    print(rd.neighbours)
    assert math.isclose(rd.cxpb + rd.mutpb + rd.mutarpb, 1), "Probabilities of operators should sum to ~1."

    print(rd)

    pop, stats, hof, logbook = main()

    final_output(hof, toolbox, logbook, pop, rd)
