#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <nesoi/kd-tree.h>
#include <nesoi/triplet-merge-tree.h>

#include "numpy-traits.h"
#include "barycenters.h"

using PyTMT  = nesoi::TripletMergeTree<float, std::uint64_t>;
using Vertex = PyTMT::Vertex;
using Degree = PyTMT::Value;

template<class T>
PyTMT build_kdistance_tree_euclidean(py::array a, size_t k)
{
    size_t n = a.shape()[0];

    PyTMT tmt(n + n * (n - 1) / 2, false);       // barycenters + all pairwise edges

    // build k-d tree
    using Traits        = NumPyTraits<T>;
    using KDTree        = nesoi::KDTree<Traits>;
    using PointHandle   = typename Traits::PointHandle;
    using DistanceType  = typename Traits::DistanceType;

    // fill point handles
    std::vector<PointHandle> handles; handles.reserve(n);
    for (size_t i = 0; i < n; ++i)
        handles.emplace_back(PointHandle {i});

    Traits traits(a);
    KDTree kdtree(traits, std::move(handles));

    // find witnessed barycenters
    BarycentersContainer<T> barycenters(n, traits.dimension());

    // find neighbors
    tmt.for_each_vertex(n, [&](Vertex u)
    {
        auto neighbors = kdtree.findK(PointHandle {u}, k);
        auto nbr_sz    = neighbors.size();

        // compute barycenter
        for (auto hd : neighbors)
            for (size_t i = 0; i < traits.dimension(); ++i)
                barycenters.coordinate(u, i) += traits.coordinate(hd.p, i) / nbr_sz;

        // compute the weight (average square distance to the defining points)
        DistanceType weight = 0;
        for (auto hd : neighbors)
        {
            for (size_t i = 0; i < traits.dimension(); ++i)
            {
                DistanceType diff = barycenters.coordinate(u,i) - traits.coordinate(hd.p, i);
                weight += diff*diff / nbr_sz;
            }
        }
        barycenters.weight(u) = -weight;
        tmt.add(u, weight);     // NB: negative of the actual weight
    });

    // build the actual tree: for every pair, add the edge as a vertex
    tmt.for_each_vertex(n, [&](Vertex u)
    {
        for (Vertex v = u + 1; v < barycenters.size(); ++v)
        {
            auto dist = barycenters.distance(u,v);
            Vertex uv = n + n*u - u*(u+1)/2 + v - 1 - u;    // vertex uv encodes the edge (u,v)
            tmt.add(uv, dist);
            tmt.merge(u, uv);
            tmt.merge(v, uv);
        }
    });

    tmt.repair();

    return tmt;
}

PyTMT build_kdistance_tree(py::array a, size_t k)
{
    if (a.ndim() == 2)
    {
        if (a.dtype().is(py::dtype::of<float>()))
            return build_kdistance_tree_euclidean<float>(a,k);
        else if (a.dtype().is(py::dtype::of<double>()))
            return build_kdistance_tree_euclidean<double>(a,k);
        else
            throw std::runtime_error("Unknown array dtype");
    } else
        throw std::runtime_error("Unknown input dimension: can only process 2D arrays");
}

void init_kdistance_tree(py::module& m)
{
    using namespace pybind11::literals;

    m.def("build_kdistance_tree",  &build_kdistance_tree,
          "data"_a, "k"_a,
          "returns the merge tree of the graph with respect to the kdistance function");
}
