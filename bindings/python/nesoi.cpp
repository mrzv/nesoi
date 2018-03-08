#include <pybind11/pybind11.h>
namespace py = pybind11;

void init_tmt(py::module&);
void init_degree_tree(py::module&);

PYBIND11_MODULE(_nesoi, m)
{
    m.doc() = "Nesoi python bindings";

    init_tmt(m);
    init_degree_tree(m);
}

