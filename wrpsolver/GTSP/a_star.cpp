#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
namespace py = pybind11;
PYBIND11_MODULE(astar, m) {
    m.def("sum_2d", [](py::array_t<double> _grid, py::array_t<int> _cityPos, py::array_t<int> _goodsClass) {
        auto grid = _grid.unchecked<2>(); // x must have ndim = 3; can be non-writeable
        auto cityPos = _cityPos.unchecked<2>(); // x must have ndim = 3; can be non-writeable
        auto goodsClass = _goodsClass.unchecked<1>(); // x must have ndim = 3; can be non-writeable
        double sum = 0;
        for (py::ssize_t i = 0; i < r.shape(0); i++)
            for (py::ssize_t j = 0; j < r.shape(1); j++)
                sum += grid(i, j);
        return r;
    });
}