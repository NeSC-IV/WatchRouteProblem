#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <math.h>
#include <unistd.h>
#include <iostream>
#include <algorithm>
#include <mutex>
#include "AStar.hpp"
#include "thread_pool.hpp"

using namespace std::placeholders;

bool AStar::Vec2i::operator == (const Vec2i& coordinates_)
{
    return (x == coordinates_.x && y == coordinates_.y);
}

AStar::Node::Node(Vec2i coordinates_, Node *parent_)
{
    parent = parent_;
    coordinates = coordinates_;
    G = H = 0;
}

AStar::uint AStar::Node::getScore()
{
    return G + H;
}

AStar::Generator::Generator()
{
    setDiagonalMovement(false);
    setHeuristic(&Heuristic::manhattan);
    direction = {
        { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 },
        { -1, -1 }, { 1, 1 }, { -1, 1 }, { 1, -1 }
    };
}

void AStar::Generator::setWorldSize(Vec2i worldSize_)
{
    worldSize = worldSize_;
}

void AStar::Generator::setDiagonalMovement(bool enable_)
{
    directions = (enable_ ? 8 : 4);
}

void AStar::Generator::setHeuristic(HeuristicFunction heuristic_)
{
    heuristic = std::bind(heuristic_, _1, _2);
}

// void AStar::Generator::addCollision(Vec2i coordinates_)
// {
//     walls.push_back(coordinates_);
// }

// void AStar::Generator::removeCollision(Vec2i coordinates_)
// {
//     auto it = std::find(walls.begin(), walls.end(), coordinates_);
//     if (it != walls.end()) {
//         walls.erase(it);
//     }
// }

// void AStar::Generator::clearCollisions()
// {
//     walls.clear();
// }
std::vector<std::vector<int>>* AStar::Generator::findPath(Vec2i source_, Vec2i target_)
{
    auto res = new std::vector<std::vector<int>>();
    if (detectCollision(source_) || detectCollision(target_)){
        std::cout << "AStar::Generator::findPath input invalid !" << std::endl;
        return res;
    }
    Node *current = nullptr;
    NodeSet openSet, closedSet;
    openSet.reserve(10000);
    closedSet.reserve(10000);
    openSet.emplace(source_.x*10000+source_.y,new Node(source_));

    while (!openSet.empty()) {
        auto current_it = openSet.begin();
        current = current_it->second;

        for (auto it = openSet.begin(); it != openSet.end(); it++) {
            auto node = it->second;
            if (node->getScore() <= current->getScore()) {
                current = node;
                current_it = it;
            }
        }

        if (current->coordinates == target_) {
            break;
        }

        // closedSet.push_back(current);
        closedSet.emplace(current->coordinates.x*10000+current->coordinates.y,current);
        openSet.erase(current_it->second->coordinates.x*10000+current_it->second->coordinates.y);

        for (uint i = 0; i < directions; ++i) {
            Vec2i newCoordinates(current->coordinates + direction[i]);
            if (detectCollision(newCoordinates) ||
                findNodeOnList(closedSet, newCoordinates)) {
                continue;
            }

            uint totalCost = current->G + ((i < 4) ? 10 : 14);

            Node *successor = findNodeOnList(openSet, newCoordinates);
            if (successor == nullptr) {
                successor = new Node(newCoordinates, current);
                successor->G = totalCost;
                successor->H = heuristic(successor->coordinates, target_);
                // openSet.push_back(successor);
                openSet.emplace(successor->coordinates.x*10000+successor->coordinates.y,successor);
                
            }
            else if (totalCost < successor->G) {
                successor->parent = current;
                successor->G = totalCost;
            }
        }
    }
    if(!(current->coordinates == target_))
        return res;

    CoordinateList path;
    std::vector<int> temp(2);
    while (current != nullptr) {
        // path.push_back(current->coordinates);
        temp[0] = current->coordinates.x;
        temp[1] = current->coordinates.y;
        res->emplace_back(temp);
        current = current->parent;
    }

    releaseNodes(openSet);
    releaseNodes(closedSet);
    return res;
}

AStar::Node* AStar::Generator::findNodeOnList(NodeSet& nodes_, Vec2i coordinates_)
{

    // for (auto node : nodes_) {
    //     if (node->coordinates == coordinates_) {
    //         return node;
    //     }
    // }
    auto iter = nodes_.find(coordinates_.x*10000+coordinates_.y);
    if (iter == nodes_.end()){
        return nullptr;
    }
    return iter->second;
}

void AStar::Generator::releaseNodes(NodeSet& nodes_)
{
    for (auto it = nodes_.begin(); it != nodes_.end();) {
        delete it->second;
        it = nodes_.erase(it);
    }
}

bool AStar::Generator::detectCollision(Vec2i coordinates_)
{
    if (coordinates_.x < 0 || coordinates_.x >= worldSize.x ||
        coordinates_.y < 0 || coordinates_.y >= worldSize.y ||
        // 0){
        (*pGrid)(coordinates_.y,coordinates_.x)==0) {
        return true;
    }
    return false;
}
void AStar::Generator::setGrid(pybind11::detail::unchecked_reference<int, 2>* _pGrid)
{
    pGrid = _pGrid;
}

AStar::Vec2i AStar::Heuristic::getDelta(Vec2i source_, Vec2i target_)
{
    return{ abs(source_.x - target_.x),  abs(source_.y - target_.y) };
}

AStar::uint AStar::Heuristic::manhattan(Vec2i source_, Vec2i target_)
{
    auto delta = std::move(getDelta(source_, target_));
    return static_cast<uint>(10 * (delta.x + delta.y));
}

AStar::uint AStar::Heuristic::euclidean(Vec2i source_, Vec2i target_)
{
    auto delta = std::move(getDelta(source_, target_));
    return static_cast<uint>(10 * sqrt(pow(delta.x, 2) + pow(delta.y, 2)));
}

AStar::uint AStar::Heuristic::octagonal(Vec2i source_, Vec2i target_)
{
    auto delta = std::move(getDelta(source_, target_));
    return 10 * (delta.x + delta.y) + (-6) * std::min(delta.x, delta.y);
}
AStar::Generator generator;

using dvector = std::vector<std::vector<int>>;
std::vector<std::vector<dvector*>> * res = nullptr;

void GetSinglePath(int i, int j){
    dvector* path = nullptr; 
    if((*res)[j][i] != 0){
        path  = new dvector(*((*res)[j][i]));
        std::reverse(path->begin(),path->end());
    }
    else{
        auto cityPos = *(generator._cityPos);
        AStar::Vec2i start({cityPos(i,0),cityPos(i,1)});
        AStar::Vec2i target({cityPos(j,0),cityPos(j,1)});
        path = generator.findPath(start,target);
        std::reverse(path->begin(),path->end());
    }
    (*res)[i][j] = path;
}

namespace py = pybind11;

PYBIND11_MODULE(Astar, m) {
    m.def("GetPath", [](py::array_t<int> _grid, py::array_t<int> _cityPos) {
        auto grid = _grid.unchecked<2>();
        auto pGrid = &grid;
        auto cityPos = _cityPos.unchecked<2>();
        int cityNum = cityPos.shape(0);

        res = new std::vector<std::vector<dvector*>>(cityNum,std::vector<dvector*>(cityNum,0));

        generator.setWorldSize({grid.shape(1), grid.shape(0)});
        generator.setHeuristic(AStar::Heuristic::euclidean);
        generator.setDiagonalMovement(true);
        generator.setGrid(pGrid);
        generator._cityPos = &cityPos;

        BS::thread_pool pool;
        bool useMultiThread = true;

        if(useMultiThread){
            for (py::ssize_t i = 0; i < cityNum; i++){
                for(py::ssize_t j = 0; j < cityNum; j++){
                    AStar::Vec2i start({cityPos(i,0),cityPos(i,1)});
                    pool.push_task(GetSinglePath,i,j);
                }
            }
            pool.wait_for_tasks();
        }
        else{
            for (py::ssize_t i = 0; i < cityNum; i++){
                std::cout << i << std::endl;
                AStar::Vec2i start({cityPos(i,0),cityPos(i,1)});
                for(py::ssize_t j = 0; j < cityNum; j++){
                    dvector* path = nullptr; 
                    if(j<i){
                        path  = new dvector(*((*res)[j][i]));
                        std::reverse(path->begin(),path->end());
                    }
                    else{
                        AStar::Vec2i end({cityPos(j,0),cityPos(j,1)});
                        path = generator.findPath(start,end);
                    }
                    (*res)[i][j] = path;
                }
            }
        }
        return res;
    },py::return_value_policy::move);

    m.def("test", []() {
        auto a = new std::vector<std::vector<dvector*>>(5,std::vector<dvector*>(5,0));
        auto pb = new dvector();
        std::vector<int> v({1,2,3,4});
        pb->emplace_back(v);
        (*a)[1][1] = pb;
        return a;
    },py::return_value_policy::move);
}