from copy import deepcopy

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph._shortest_path import shortest_path
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import pairwise_distances


def globaljudge(X, Y, K):
    """
    Please cite: "Deyu Meng, Yee Leung, Zongben Xu.
    Evaluating nonlinear dimensionality reduction based on its local
    and global quality assessments, Neurocomputing, 2011, 74(6): 941-948."

    The code is written by Deyu Meng & adapated by Andrew Lensen

    :param X: Original data (feature-major: dim*num)
    :param Y: Embedding data (feature-major: dim*num)
    :param K: Neighborhood size
    """
    i_major = X.T
    D = pairwise_distances(i_major)
    N = D.shape[0]
    # ind = np.argsort(D, axis=1)
    # # set all non-nns as inf dist
    # for i in range(N):
    #     # includes itself so K+1
    #     D[i, ind[i, K + 1:]] = 0.#np.inf
    #
    # np.fill_diagonal(D,0.)
#    E = (D != np.inf)  # Edge information for subsequent graph overlay
    csr_graph = csr_matrix(D)
    sp = shortest_path(csr_graph)
    centre_idx = np.argmin(np.max(sp,axis=1))

    #geodesic distance...maybe it is already tho


if __name__ == '__main__':
    # GENERATE SAMPLED DATA
    angle = np.pi * (1.5 * np.random.rand(1, 600) - 1)
    height = 5 * np.random.rand(1, 600)
    X1 = np.vstack((np.cos(angle), height, np.sin(angle)))

    angle = np.pi * (1.5 * np.random.rand(1, 100) - 1)
    height = 5 * np.random.rand(1, 100)
    X2 = np.vstack((-np.cos(angle), height, 2 - np.sin(angle)))

    X = np.hstack((X1, X2))
    print(X)
    Y_isomap = Isomap(n_neighbors=6).fit_transform(deepcopy(X.T))
    Y_pca = PCA(n_components=2).fit_transform(deepcopy(X.T))
    G_isomap = globaljudge(X, Y_isomap, 6)
    G_pca = globaljudge(X, Y_pca, 6)
    print(G_isomap)
    print(G_pca)
