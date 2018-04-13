#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "tmt.h"

void init_degree_tree(py::module&);
void init_kdistance_tree(py::module&);

PYBIND11_MODULE(_nesoi, m)
{
    m.doc() = "Nesoi python bindings";

    init_tmt<std::uint32_t, std::uint32_t>(m, "<uint32>");
    init_degree_tree(m);

    init_tmt<float, std::uint64_t>(m, "<float>");
    init_kdistance_tree(m);
}

