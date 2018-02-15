import numpy as np
import nesoi

points = np.concatenate([np.random.normal(loc = [2., 2.],  size = (100,2)),
                         np.random.normal(loc = [-2., 2.], size = (100,2)),
                         np.random.normal(loc = [0., -2.], size = (100,2))])


import matplotlib.pyplot as plt

plt.scatter([p[0] for p in points], [p[1] for p in points])
plt.show()

tmt = nesoi.build_tree(points, .7)
nesoi.plot.plot_bars(tmt, show = True)
nesoi.plot.plot_diagram(tmt, show = True)
