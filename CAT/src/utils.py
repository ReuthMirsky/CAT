import numpy as np
from heapq import heappush, heappop
from src.environment import ToolFetchingEnvironment
from itertools import count
from collections import Iterable


class Point2D:
    def __init__(self, p):
        self._x = p[0]
        self._y = p[1]

    def __iter__(self):
        yield self._x
        yield self._y


    def __getitem__(self, i):
        assert i==0 or i==1
        if i==0:
            return self._x
        return self._y

    def __hash__(self):
        return hash((self._x, self._y))

    def __eq__(self, other):
        return self._x == other[0] and self._y == other[1]

    __req__ = __eq__

    def __add__(self, other):
        return Point2D([self._x+other[0], self._y+other[1]])

    def __sub__(self, other):
        return Point2D([self._x-other[0], self._y-other[1]])

    def __mul__(self, other):
        if hasattr(other, '__getitem__'):
            return Point2D([self._x*other[0], self._y*other[1]])
        return Point2D([self._x*other, self._y*other])

    __rmul__ = __mul__

    def __truediv__(self, other):
        if hasattr(other, '__getitem__'):
            return Point2D([self._x/other[0], self._y/other[1]])
        return Point2D([self._x/other, self._y/other])

    def __floordiv__(self, other):
        if hasattr(other, '__getitem__'):
            return Point2D([self._x//other[0], self._y//other[1]])
        return Point2D([self._x//other, self._y//other])

    def __isub__(self, other):
            self._x -= other[0]
            self._y -= other[1]

    def __iadd__(self, other):
        self._x += other[0]
        self._y += other[1]

    def __imul__(self, other):
        if hasattr(other, '__getitem__'):
            self._x *= other[0]
            self._y *= other[1]
        else:
            self._x *= other
            self._y *= other


    def __idiv__(self, other):
        if hasattr(other, '__getitem__'):
            self._x /= other[0]
            self._y /= other[1]
        else:
            self._x /= other
            self._y /= other

    def __ifloordiv__(self, other):
        if hasattr(other, '__getitem__'):
            self._x //= other[0]
            self._y //= other[1]
        else:
            self._x //= other
            self._y //= other


    def __str__(self):
        return f'({self._x}, {self._y})'

    __repr__ = __str__

    def __deepcopy__(self, memo):
        return Point2D([self._x, self._y])

    def __copy__(self):
        return Point2D([self._x, self._y])

    def __lt__(self, other):
        return (self._x**2 + self._y**2) < (other._x**2 + other._y**2)
        

def dist(src, goal):
    return abs(src[0]-goal[0]) + abs(src[1]-goal[1])

def horz_dist(src, goal):
    return 0.9*abs(src[0]-goal[0]) + abs(src[1]-goal[1])

def vert_dist(src, goal):
    return abs(src[1]-goal[1])

def gen_graph(obs, width, height):
    def graph(node):
        deltas = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        neighbors = []
        for d in deltas:
            new = node+d
            if new not in obs and new[0] >= 0 and new[0] < width and new[1] >= 0 and new[1] < height:
                neighbors.append((1,new))
        return neighbors
    return graph


def astar(graph, start, finish, h=dist):
    q = [(0,0,start,[])]
    visited = set()
    while len(q) > 0:
        d, val, node, path = heappop(q)
        path.append(node)
        if np.array_equal(node, finish):
            return path
        if node in visited:
            continue
        visited.add(node)
        for cost, n in graph(node):
            heappush(q, (val+cost+h(n, finish), val+cost, n, list(path)))


def first_ind(l):
    for i,e in enumerate(l):
        if e:
            return i
    return None
