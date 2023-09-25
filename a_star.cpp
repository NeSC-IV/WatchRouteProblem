#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
namespace py = pybind11;
typedef struct buffer_path{
    std::vector<std::vector<int>> distance;
    std::vector<std::vector<int>> path;
    buffer_path(){}
    buffer_path(int num){
        std::vector<std::vector<int>> temp(num,std::vector<int>(num));
        distance = temp;
    }
}path;
PYBIND11_MODULE(astar, m) {
    // m.def("sum_2d", [](py::array_t<float> _grid, py::array_t<int> _cityPos, py::array_t<int> _goodsClass) {
    m.def("sum_2d", [](py::array_t<float> _grid, py::array_t<int> _cityPos, py::array_t<int> _goodsClass) {
        auto grid = _grid.unchecked<2>();
        auto cityPos = _cityPos.unchecked<2>();
        auto goodsClass = _goodsClass.unchecked<1>();
        int cityNum = cityPos.shape(0);
        path res(cityNum);
        py::object obj = py::cast(res);
        return obj;
    });
}