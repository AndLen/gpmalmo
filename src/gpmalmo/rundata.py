import cachetools
from defaultlist import defaultlist

data = None
data_t = None
labels = None
outdir = None
pairwise_distances = None
ordered_neighbours = None
neighbours = None
all_orderings = None
identity_ordering = None
nobj = 2
fitnessCache = defaultlist(lambda: cachetools.LRUCache(maxsize=1e6))
accesses = 0
stores = 0

max_depth = 8#7#12#8
max_height = 14#10#17#14
pop_size = 100#1024#100
cxpb = 0.7
mutpb = 0.15
mutarpb = 0.15
num_trees = 34
gens = 1000

num_instances = 0
num_features = 0

