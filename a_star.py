
import math
import shapely
from astar import AStar
from _global import *


class AStarSolver(AStar):

    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    def __init__(self, polygon):
        self.polygon = polygon
        # self.polygon = polygon.buffer(zoomRate/1000)
        self.step = (zoomRate/100)

    def isReachable(self, point):
        return self.polygon.buffer(self.step).covers(shapely.Point(point))

    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        (x1, y1) = n1
        (x2, y2) = n2
        return math.hypot(x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adajcent"""
        return self.step

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """

        x, y = node
        neighborList = [(x+self.step, y), (x-self.step, y),
                        (x, y+self.step), (x, y-self.step)]
        return [neighbor for neighbor in neighborList if self.isReachable(neighbor)]


def findPath(start, goal, freeSpace):

    # let's solve it
    aStarSolver = AStarSolver(freeSpace)
    start = (int(start[0]/(zoomRate/100)) * (zoomRate/100), int(start[1]/(zoomRate/100)) * (zoomRate/100))
    goal = (int(goal[0]/(zoomRate/100)) * (zoomRate/100), int(goal[1]/(zoomRate/100)) * (zoomRate/100))
    if (start == goal):
        return [start, goal], 0

        
    foundPath = list(aStarSolver.astar(start, goal))
    distance = len(foundPath) * aStarSolver.step

    return foundPath, distance
