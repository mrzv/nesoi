import numpy as np
import nesoi

points = np.concatenate([np.random.normal(loc = [2., 2.],  size = (100,2)),
                         np.random.normal(loc = [-2., 2.], size = (100,2)),
                         np.random.normal(loc = [0., -2.], size = (100,2))])


import matplotlib.pyplot as plt

ks = []
ps = []
for k in range(1, 100):
    tmt = nesoi.build_kdistance_tree(points, k)
    triplets = tmt.traverse_persistence()
    ks += [k for x,y,z in triplets]
    ps += [tmt.value(y) - tmt.value(x) for x,y,z in triplets]

plt.scatter(ks, ps)
plt.show()
