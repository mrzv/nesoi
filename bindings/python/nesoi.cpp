#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <nesoi/triplet-merge-tree.h>
#include <nesoi/kd-tree.h>

#include "numpy-traits.h"

using PyTMT  = nesoi::TripletMergeTree<std::uint32_t, std::uint32_t>;
using Vertex = PyTMT::Vertex;
using Degree = PyTMT::Value;

template<class T>
struct ExplicitDistances
{
    using IndexType     = Vertex;
    using DistanceType  = float;

           ExplicitDistances(const py::array& a_):
               a(a_), n(static_cast<size_t>(1 + std::sqrt(1 + 8*a.shape(0))/2))        {}

    DistanceType operator()(size_t u, size_t v) const
    {
        if (u == v)
            return 0;

        if (u < v)              // below u must be larger than v
            std::swap(u,v);

        size_t idx = n*v - v*(v+1)/2 + u - 1 - v;
        const void* xptr = a.data(idx);
        T x = *static_cast<const T*>(xptr);

        return x;
    }

    IndexType   begin() const       { return 0; };
    IndexType   end() const         { return n; }
    IndexType   size() const        { return end() - begin(); }

    const py::array&    a;
    size_t              n;
};

template<class T>
PyTMT build_tree_euclidean(py::array a, double eps)
{
    size_t n = a.shape()[0];

    PyTMT tmt(n, true);

    // build k-d tree
    using Traits        = NumPyTraits<T>;
    using KDTree        = nesoi::KDTree<Traits>;
    using PointHandle   = typename Traits::PointHandle;

    // fill point handles
    std::vector<PointHandle> handles; handles.reserve(n);
    for (size_t i = 0; i < n; ++i)
        handles.emplace_back(PointHandle {i});

    Traits traits(a);
    KDTree kdtree(traits, std::move(handles));

    // find neighbors
    tmt.for_each_vertex([&](Vertex u)
                        {
                            auto neighbors = kdtree.findR(PointHandle {u}, eps);

                            tmt.add(u, neighbors.size() - 1);       // -1 for u itself

                            for (auto hd : neighbors)
                            {
                                Vertex v = traits.id(hd.p);
                                if (u != v && tmt.contains(v))
                                    tmt.merge(u,v);
                            }
                        });
    tmt.repair();

    return tmt;
}

template<class T>
PyTMT build_tree_explicit(py::array a, double eps)
{
    using Distances = ExplicitDistances<T>;

    Distances distances(a);
    PyTMT     tmt(distances.size(), true);

    tmt.for_each_vertex([&](Vertex u)
                        {
                            Degree degree = 0;
                            auto   sz     = tmt.size();
                            for (Vertex v = 0; v < sz; ++v)
                                if (distances(u,v) <= eps)
                                    ++degree;
                            degree -= 1;        // for u itself
                            tmt.add(u, degree);

                            for (Vertex v = 0; v < sz; ++v)
                                if (u != v && distances(u,v) <= eps && tmt.contains(v))
                                    tmt.merge(u,v);
                        });
    tmt.repair();

    return tmt;
}

PyTMT build_tree(py::array a, double eps)
{
    if (a.ndim() == 2)
    {
        if (a.dtype().is(py::dtype::of<float>()))
            return build_tree_euclidean<float>(a,eps);
        else if (a.dtype().is(py::dtype::of<double>()))
            return build_tree_euclidean<double>(a,eps);
        else
            throw std::runtime_error("Unknown array dtype");
    } else if (a.ndim() == 1)
    {
        if (a.dtype().is(py::dtype::of<float>()))
            return build_tree_explicit<float>(a,eps);
        else if (a.dtype().is(py::dtype::of<double>()))
            return build_tree_explicit<double>(a,eps);
        else
            throw std::runtime_error("Unknown array dtype");
    } else
        throw std::runtime_error("Unknown input dimension: can only process 1D and 2D arrays");
}

std::map<Vertex, std::vector<Vertex>>
clusters(const PyTMT& tmt, Degree k)
{
    std::vector<Vertex> s;
    std::unordered_map<Vertex, Vertex> rep;
    for (Vertex u = 0; u < tmt.size(); ++u)
    {
        Vertex x = u;
        if (tmt.value(x) < k)
            continue;

        while (rep.find(x) == rep.end())
        {
            s.push_back(x);
            Vertex s = tmt[x].through;
            Vertex v = tmt[x].to;

            if (x == v || tmt.value(s) < k)
                break;
            else
                x = v;
        }

        if (s.empty())
            continue;

        Vertex v = x;
        auto it = rep.find(x);
        if (it != rep.end())
            v = it->second;

        for (Vertex x : s)
            rep[x] = v;

        s.clear();
    }

    std::map<Vertex, std::vector<Vertex>> clusters;
    for (auto& x : rep)
        clusters[x.second].push_back(x.first);

    return clusters;
}

PYBIND11_MODULE(_nesoi, m)
{
    m.doc() = "Nesoi python bindings";

    using namespace pybind11::literals;

    py::class_<PyTMT>(m, "TMT", "triplet merge tree")
        .def("__len__",         &PyTMT::size,               "size of the tree")
        .def("__contains__",    &PyTMT::contains,           "test whether the tree contains the vertex")
        .def("merge",           [](PyTMT& tmt, Vertex u, Vertex v)
                                {
                                    tmt.merge(u, v);
                                },                          "merge two vertices (i.e., add an edge to the domain)")
        .def("representative",  &PyTMT::representative,     "find representative of a node at a given level")
        .def("value",           &PyTMT::value,              "function value of the given vertex")
        .def("__repr__",        [](const PyTMT& tmt)    { std::ostringstream oss; oss << "Tree with " << tmt.size() << " nodes"; return oss.str(); })
        .def("traverse_persistence",    [](const PyTMT& tmt)
                                        {
                                            std::vector<std::tuple<Vertex, Vertex, Vertex>> result;
                                            tmt.traverse_persistence([&result](Vertex u, Vertex s, Vertex v) { result.emplace_back(u,s,v); });
                                            return result;
                                        },  "traverse persistence, return list of vertex triplets")
        .def("clusters",        &clusters, "k"_a, "find all clusters at the given degree threshold")
    ;

    m.def("build_tree",  &build_tree,
          "data"_a, "eps"_a,
          "returns the merge tree of the graph with respect to the degree function");
}

