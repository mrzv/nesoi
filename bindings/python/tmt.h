#pragma once

#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
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

    std::string classname = "TMT" + suffix;
    py::class_<PyTMT>(m, classname.c_str(), "triplet merge tree")
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
        .def("clusters",        &clusters<PyTMT>, "k"_a, "find all clusters at the given threshold")
    ;
}
