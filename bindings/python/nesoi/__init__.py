from __future__ import absolute_import
from ._nesoi import *
from .plot   import *

def intervals(tmt):
    triplet_values = plot._triplet_values(tmt)
    triplet_values.sort(reverse = True)
    return [(u, birth, death) for (persistence, birth, death, _, u) in triplet_values]
