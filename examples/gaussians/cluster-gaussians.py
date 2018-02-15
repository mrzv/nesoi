import numpy as np
import nesoi

points = np.concatenate([np.random.normal(loc = [2., 2.],  size = (100,2)),
                         np.random.normal(loc = [-2., 2.], size = (100,2)),
                         np.random.normal(loc = [0., -2.], size = (100,2))])


import matplotlib.pyplot as plt

#plt.scatter([p[0] for p in points], [p[1] for p in points])
#plt.show()

tmt = nesoi.build_tree(points, .7)
#centers = nesoi.plot_bars(tmt, show = True)
#print(centers)

for (u,b,d) in nesoi.intervals(tmt):
    print(f"Cluster with center {u:3} is present for k in [{b},{d}]")

print()
print("At k = 12, clusters have the following sizes:")
clusters = tmt.clusters(k = 12)
for k,v in clusters.items():
    print(f"Cluster with center {k:3} has size {len(v)}")
