
import math
import shapely
from . import AStar
from ...Global import *


class AStarSolver(AStar):

    """sample use of the astar algorithm. In this exemple we work on a maze made of ascii characters,
    and a 'node' is just a (x,y) tuple that represents a reachable position"""

    def __init__(self, polygon):
        self.step = (zoomRate/50)
        self.polygon = polygon
        # self.polygon = polygon.buffer(zoomRate/500)

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
        return self.step

    def neighbors(self, node):
        """ for a given coordinate in the maze, returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """

        x, y = node
        neighborList = [(x+self.step, y), (x-self.step, y),
                        (x, y+self.step), (x, y-self.step)]
        return [neighbor for neighbor in neighborList if self.isReachable(node,neighbor)]


def findPath(start, goal, freeSpace):

    # let's solve it
    aStarSolver = AStarSolver(freeSpace)
    step = aStarSolver.step
    start = (int(start[0]/(step)) * (step), int(start[1]/(step)) * (step))
    goal = (int(goal[0]/(step)) * (step), int(goal[1]/(step)) * (step))
    if (start == goal):
        return [start, goal], 0
        
    result = aStarSolver.astar(start, goal)
    foundPath = list(result)
    distance = len(foundPath) * aStarSolver.step

    return foundPath, distance
