import numpy as np
from numba import jit

from gpmalmo import rundata
from gptools.array_wrapper import ArrayWrapper
from gptools.gp_util import evaluateTrees
from gptools.util import cachedError


def evalGPMalNC(data_t, toolbox, individual):
    dat_array = evaluateTrees(data_t, toolbox, individual)

    hashable = ArrayWrapper(dat_array)
    # in [-1,1]
    # TODO: need to properly consider the situation where there are duplicate ith-nearest neighbours...
    # At the moment, if we don't do something like this, it likes to find a dumb optima where all distances are ~~0
    args = (rundata.all_orderings, rundata.identity_ordering, dat_array)
    cost, ratio_uniques = cachedError(hashable, eval_similarity_st, rundata, args=args, kargs={}, index=0)

    num_trees = len(individual)
    if ratio_uniques < 0.9:
        # lower ratio is worse, so higher return value
        # 2- so that always worse than a valid soln
        return 2 - ratio_uniques, num_trees

    # reshape to be in [0,2] and then [0,1]
    to_return = (-cost + 1) / 2, num_trees
    if to_return[0] == 0:
        print("wow")
    return to_return


@jit(nopython=True)
def spearmans(o1, o2):
    d = np.abs(o1 - o2) ** 2
    n = len(o1)
    rho = 1 - (6 * d.sum() / (n * ((n ** 2) - 1)))
    return rho


# TODO: dynamic length
# identity_ordering = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


@jit(nopython=True)
def eval_similarity(orderings, identity_ordering, dat_array):
    sum = 0.
    n_instances = orderings.shape[0]
    n_neighbours = orderings.shape[1]

    # arr = deepcopy(dat_array)
    num_uniques = 0
    for i in range(n_instances):
        pair_dists = np.empty((n_neighbours,), dtype=np.double)
        array_a = dat_array[i]
        array_b = dat_array[orderings[i]]
        # for j in range(len(array_a)):
        for j in range(n_neighbours):
            pair_dists[j] = np.linalg.norm(array_a - array_b[j])
        # pair_dists = cdist(dat_array[i].reshape(1, -1), dat_array[orderings[i]])[0]

        # if pair_dists.min == pair_dists.max:
        #    return -1
        distincts = len(np.unique(pair_dists))
        num_uniques = num_uniques + (distincts / n_neighbours)
        argsort = np.argsort(pair_dists)
        # rho = spearmanr(identity_ordering, argsort).correlation
        # print(rho)
        rho = spearmans(identity_ordering, argsort)
        # do we need to transform it here...or does it not matter
        sum = sum + rho
        # print(argsort)
    # print (num_uniques/n_instances)
    return sum / n_instances, num_uniques / n_instances


@jit(nopython=True)
def eval_similarity_st(orderings, identity_ordering, dat_array):
    return eval_similarity(orderings, identity_ordering, dat_array)
