#pragma once

#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <nesoi/triplet-merge-tree.h>

template<class PyTMT>
std::map<typename PyTMT::Vertex, std::vector<typename PyTMT::Vertex>>
clusters(const PyTMT& tmt, typename PyTMT::Value k)
{
    using Vertex = typename PyTMT::Vertex;

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

template<class Value_, class Vertex_>
void init_tmt(py::module& m, std::string suffix)
{
    using namespace pybind11::literals;

    using PyTMT  = nesoi::TripletMergeTree<Value_, Vertex_>;
    using Vertex = typename PyTMT::Vertex;
    using Value  = typename PyTMT::Value;
    using EdgeVector = typename std::vector<std::tuple<Vertex, Vertex>>;

    std::string classname = "TMT" + suffix;
    py::class_<PyTMT>(m, classname.c_str(), "triplet merge tree")
        .def(py::init<size_t, bool>())
        .def("__len__",         &PyTMT::size,               "size of the tree")
        .def("__contains__",    &PyTMT::contains,           "test whether the tree contains the vertex")
        .def("__getitem__",     [](PyTMT& tmt, Vertex u)
                                {
                                    auto e = tmt[u];
                                    return std::make_pair(e.through, e.to);
                                },                          "return the label on the edge out of e")
        .def("add",             &PyTMT::add,                "add a vertex to the tree")
        .def("link",            &PyTMT::link,               "add a triplet (u,s,v) to the tree")
        .def("merge",           [](PyTMT& tmt, Vertex u, Vertex v)
                                {
                                    tmt.merge(u, v);
                                },                          "merge two vertices (i.e., add an edge to the domain)")
        .def("repair",          [](PyTMT& tmt)
                                {
                                    tmt.repair();
                                },                          "repair the tree after a sequence of merges")
        .def("representative",  &PyTMT::representative,     "find representative of a node at a given level")
        .def("value",           &PyTMT::value,              "function value of the given vertex")
        .def("__repr__",        [](const PyTMT& tmt)    { std::ostringstream oss; oss << "Tree with " << tmt.size() << " nodes"; return oss.str(); })
        .def("traverse_persistence",    [](const PyTMT& tmt)
                                        {
                                            std::vector<std::tuple<Vertex, Vertex, Vertex>> result;
                                            tmt.traverse_persistence([&result](Vertex u, Vertex s, Vertex v) { result.emplace_back(u,s,v); });
                                            return result;
                                        },  "traverse persistence, return list of vertex triplets")
        .def("clusters",        &clusters<PyTMT>, "k"_a, "find all clusters at the given threshold")
        .def("compute_mt",      [](PyTMT& tmt, const EdgeVector& edges, py::array_t<Value> values, bool negate)
                                {
                                    tmt.set_negate(negate);
                                    py::buffer_info buf = values.request();
                                    if (buf.ndim != 1)
                                        throw std::runtime_error("Expected 1D array");
                                    if (buf.size != edges.size())
                                        throw std::runtime_error("Array size and graph size do not match.");
                                    Value* val_ptr = (Value*) (values.ptr());
                                    tmt.compute_mt(edges, val_ptr, negate);
                                }, "compute merge tree")
        .def("simplify",        [](PyTMT& tmt,  const EdgeVector& edges, py::array_t<Value> values, Value epsilon, bool negate)
                                {
                                    py::buffer_info buf = values.request();
                                    if (buf.ndim != 1) throw std::runtime_error("Expected 1D array");
                                    Value* val_ptr = (Value*) (buf.ptr);
                                    return tmt.simplify(edges, val_ptr, epsilon, negate);
                                }, "simplify function on graph")
        .def("simplify_ls",     [](PyTMT& tmt,  const EdgeVector& edges, py::array_t<Value> values, Value epsilon, Value level_value, bool negate)
                                {
                                    py::buffer_info buf = values.request();
                                    if (buf.ndim != 1) throw std::runtime_error("Expected 1D array");

                                    Value* val_ptr = (Value*) (buf.ptr);

                                    return tmt.simplify(edges, val_ptr, epsilon, level_value, negate);
                                }, "simplify level set of function on graph")
        .def_property_readonly("negate", &PyTMT::negate,    "indicates whether the tree follows super- or sub-levelsets")
        .def(py::pickle(
            [](const PyTMT& tmt)        // __getstate__
            {
                bool negate = tmt.negate();
                std::vector<std::tuple<Vertex,Value>>           vertex_values;
                std::vector<std::tuple<Vertex,Vertex,Vertex>>   triplets;
                for (Vertex u = 0; u < tmt.size(); ++u)
                {
                    vertex_values.emplace_back(u, tmt.value(u));
                    auto e = tmt[u];
                    triplets.emplace_back(u, e.through, e.to);
                }
                return py::make_tuple(negate, vertex_values, triplets);
            },
            [](py::tuple t)              // __setstate__
            {
                if (t.size() != 3)
                    throw std::runtime_error("Invalid state!");

                std::vector<std::tuple<Vertex,Value>>           vertex_values = t[1].cast<std::vector<std::tuple<Vertex,Value>>>();
                std::vector<std::tuple<Vertex,Vertex,Vertex>>   triplets      = t[2].cast<std::vector<std::tuple<Vertex,Vertex,Vertex>>>();
                PyTMT tmt(vertex_values.size(), t[0].cast<bool>());

                for (auto& x : vertex_values)
                    tmt.add(std::get<0>(x), std::get<1>(x));

                for (auto& x : triplets)
                    tmt.link(std::get<0>(x), std::get<1>(x), std::get<2>(x));

                return tmt;
            }))
    ;
}
