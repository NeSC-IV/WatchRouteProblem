
import math
import shapely
from . import AStar
from ...Global import *


class AStarSolver(AStar):

    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    def __init__(self, polygon):
        self.step = (zoomRate/100)
        self.polygon = polygon

    def isReachable(self,start, point):
        line = shapely.LineString([start,point])
        return self.polygon.contains(line)

    def heuristic_cost_estimate(self, n1, n2):
        """computes the 'direct' distance between two (x,y) tuples"""
        (x1, y1) = n1
        (x2, y2) = n2
        return math.hypot(x2 - x1, y2 - y1)

    def distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adajcent"""
        (x1, y1) = n1
        (x2, y2) = n2
        return math.hypot(x2 - x1, y2 - y1)

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """

        x, y = node
        step = self.step
        neighborList = [
                        (x+step, y), (x-step, y),
                        (x, y+step), (x, y-step),
                        (x-step,y-step),(x+step,y+step),
                        ]
        return [neighbor for neighbor in neighborList if self.isReachable(node,neighbor)]

    def is_goal_reached(self, current, goal) -> bool:
        (x1, y1) = current
        (x2, y2) = goal
        return (math.hypot(x2 - x1, y2 - y1)) < self.step

def findPath(start, goal, freeSpace):

    # let's solve it
    aStarSolver = AStarSolver(freeSpace)
    result = aStarSolver.astar(start, goal)
    foundPath = list(result)
    distance = len(foundPath) * aStarSolver.step

    return foundPath, distance
