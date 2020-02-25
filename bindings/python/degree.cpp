#include <cmath>
#include <chrono>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <nesoi/kd-tree.h>
#include <nesoi/triplet-merge-tree.h>

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
PyTMT build_degree_tree_euclidean(py::array a, double eps)
{
    size_t n = a.shape()[0];

    PyTMT tmt(n, true);

    using clock = std::chrono::steady_clock;
    using sec = std::chrono::duration<double>;

    auto start = clock::now();

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

    std::cerr << "Time to construct k-d tree: " << sec(clock::now() - start).count() << " seconds" << std::endl;
    start = clock::now();

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

    std::cerr << "Time to build TMT: " << sec(clock::now() - start).count() << " seconds" << std::endl;

    return tmt;
}

template<class T>
PyTMT build_degree_tree_explicit(py::array a, double eps)
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

PyTMT build_degree_tree(py::array a, double eps)
{
    if (a.ndim() == 2)
    {
        if (a.dtype().is(py::dtype::of<float>()))
            return build_degree_tree_euclidean<float>(a,eps);
        else if (a.dtype().is(py::dtype::of<double>()))
            return build_degree_tree_euclidean<double>(a,eps);
        else
            throw std::runtime_error("Unknown array dtype");
    } else if (a.ndim() == 1)
    {
        if (a.dtype().is(py::dtype::of<float>()))
            return build_degree_tree_explicit<float>(a,eps);
        else if (a.dtype().is(py::dtype::of<double>()))
            return build_degree_tree_explicit<double>(a,eps);
        else
            throw std::runtime_error("Unknown array dtype");
    } else
        throw std::runtime_error("Unknown input dimension: can only process 1D and 2D arrays");
}


void init_degree_tree(py::module& m)
{
    using namespace pybind11::literals;

    m.def("build_degree_tree",  &build_degree_tree,
          "data"_a, "eps"_a,
          "returns the merge tree of the graph with respect to the degree function");
}
