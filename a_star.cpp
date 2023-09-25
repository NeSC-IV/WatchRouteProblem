#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <utility>
#include <vector>
#include <typeinfo>
namespace py = pybind11;
PYBIND11_MODULE(astar, m) {
    m.def("sum_2d", [](py::array_t<int> _grid, py::array_t<int> _cityPos, py::array_t<int> _goodsClass) {
        auto grid = _grid.unchecked<2>();
        pybind11::detail::unchecked_reference<int, 2>* pGrid = &grid;
        std::cout << (*pGrid)(1,1) << std::endl;

        // std::cout << (*pGrid) << std::endl;
        auto cityPos = _cityPos.unchecked<2>();
        auto goodsClass = _goodsClass.unchecked<1>();
        int cityNum = cityPos.shape(0);
        std::vector<std::vector<int>> res;
        return res;
    });
}