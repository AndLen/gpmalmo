# GM.m
from copy import deepcopy

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances


def distance(a,b):
    return np.sqrt((a*a + b*b)-(2*a*b))

def globaljudge(X, Y, K):
    """
    Please cite: "Deyu Meng, Yee Leung, Zongben Xu.
    Evaluating nonlinear dimensionality reduction based on its local
    and global quality assessments, Neurocomputing, 2011, 74(6): 941-948."

    The code is written by Deyu Meng & adapated by Andrew Lensen

    :param X: Original data (dim*num)
    :param Y: Embedding data (dim*num)
    :param K: Neighborhood size
    """

    D = pairwise_distances(X.T)
    print(D)
    #print(np.diag(D))
    N = D.shape[0]
    #INF = 1000 * np.amax(D) * N_dim  # effectively infinite distance
    INF = np.inf
    ind = np.argsort(D,axis=1)
    for ii in range(N):
        D[ii, ind[K:, ii]] = INF
    print(D)
    D = np.minimum(D, D.T)  # Make sure distance matrix is symmetric
    E = (1 - (D == INF))  # Edge information for subsequent graph overlay

    PP = np.empty((N, N), dtype=list)
    for ii in range(N):
        for jj in range(N):
            PP[ii, jj] = [ii, jj]

    # calculate shortest paths
    for k in range(N):
        for ii in range(N):
            for jj in range(N):
                if D[ii, jj] > D[ii, k] + D[k, jj]:
                    PP[ii, jj] = [PP[ii, k][-2]] + PP[k, jj]
        D = np.minimum(D, np.tile(D[:, k], (N,1)) + np.tile(D[k, :], (N, 1)))

    DX = deepcopy(D)
    # TODO: ???
    k = max(D)
    k, StartP = D.min(0), D.argmin(0)
    StartP = np.argsort(k)
    MD = max(D[StartP, :])

    Sequence = list(range(N))
    Sequence[(Sequence == StartP).nonzero()] = []
    # GM.m:54
    # Sequence[find(Sequence == StartP)] = []
    # GM.m:55
    Leaf = []
    # GM.m:56
    while len(Sequence) > 0:
        Po = Sequence[0]
        TSe = PP[StartP, Po]
        Leaf = [Leaf, Po]
        TSe[np.nonzero(TSe == StartP)] = []
        while len(TSe) > 1:
            Sequence[np.nonzero(Sequence == TSe(0))] = []
            Leaf[np.nonzero(Leaf == TSe(0))] = []
            TSe[np.nonzero(TSe == TSe(0))] = []

        Sequence[np.nonzero(Sequence == TSe(0))] = []
        TSe[np.nonzero(TSe == TSe(0))] = []

    NLeaf = len(Leaf)
    Leaf2 = []
    FinalLeaf = []
    MaxD = []
    k = 0
    # GM.m:77
    for ii in range(NLeaf):
        TL = Leaf[ii]
        TempLeaf2 = PP[StartP, TL][2]
        if np.nonzero(Leaf2 == TempLeaf2) > 0:
            Tempi = np.nonzero(Leaf2 == TempLeaf2)
            if D[StartP, TL] > D[StartP, FinalLeaf[Tempi]]:
                FinalLeaf[Tempi] = TL
                MaxD[Tempi] = D[StartP, TL]
        else:
            if D[StartP, TL] > MD / 6:
                k = k + 1
                Leaf2[k] = TempLeaf2
                FinalLeaf[k] = TL
                MaxD[k] = D[StartP, TL]

    ########################3

    SequenceX = DX[StartP, FinalLeaf]
    DY = distance(Y, Y)
    SequenceY = DY[StartP, FinalLeaf]

    Global_ind = (1 + spearmanr(SequenceX.T, SequenceY.T)[0]) / 2
    print(Global_ind)
    return Global_ind


if __name__ == '__main__':
    # GENERATE SAMPLED DATA
    angle = np.pi * (1.5 * np.random.rand(1, 600) - 1)
    height = 5 * np.random.rand(1, 600)
    X1 = np.vstack((np.cos(angle), height, np.sin(angle)))

    angle = np.pi * (1.5 * np.random.rand(1, 100) - 1)
    height = 5 * np.random.rand(1, 100)
    X2 = np.vstack((-np.cos(angle), height, 2 - np.sin(angle)))

    X = np.hstack((X1,X2))
    print(X)
    Y_isomap = Isomap(n_neighbors=6).fit_transform(deepcopy(X.T))
    Y_pca = PCA(n_components=2).fit_transform(deepcopy(X.T))
    G_isomap = globaljudge(X, Y_isomap, 6)
    G_pca = globaljudge(X,Y_pca,6)
    print(G_isomap)
    print(G_pca)

