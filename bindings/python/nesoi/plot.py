def _triplet_values(tmt):
    triplets = tmt.traverse_persistence()
    triplet_values = []
    for (u,s,v) in triplets:
        if u != v:
            if tmt.value(u) != tmt.value(s):        # skip 0-persistence
                triplet_values.append((tmt.value(u) - tmt.value(s), tmt.value(u), tmt.value(s), tmt.value(v)))
        else:
            if tmt.value(u) != 0:
                triplet_values.append((tmt.value(u), tmt.value(u), 0, tmt.value(v)))
    return triplet_values

def plot_bars(tmt, show = False):
    """Plot the barcode."""

    import matplotlib.pyplot as plt

    triplet_values = _triplet_values(tmt)
    triplet_values.sort(reverse = True)

    for i,x in enumerate(triplet_values):
        plt.plot([x[2], x[1]], [i,i], color = 'b')

    if show:
        plt.show()

def plot_diagram(tmt, show = False):
    """Plot the persistence diagram."""

    import matplotlib.pyplot as plt

    triplet_values = _triplet_values(tmt)

    min_birth = min(p[2] for p in triplet_values)
    max_birth = max(p[2] for p in triplet_values)
    min_death = min(p[1] for p in triplet_values)
    max_death = max(p[1] for p in triplet_values)

    plt.axes().set_aspect('equal', 'datalim')

    min_diag = min(min_birth, min_death)
    max_diag = max(max_birth, max_death)

    plt.scatter([p[2] for p in triplet_values], [p[1] for p in triplet_values])
    plt.plot([min_diag, max_diag], [min_diag, max_diag])        # diagonal

    if show:
        plt.show()

