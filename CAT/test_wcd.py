import numpy as np
from src import wcd_utils
epsilon = 1/11
def h(s, sp):
    p1,p2 = s
    (x1,y1),(x2,y2) = p1,p2
    p1p,p2p = sp
    (x1p,y1p),(x2p,y2p) = p1p,p2p
    if p1 == p2:
        return (2-epsilon)*max(abs(x1-x1p)+abs(y1-y1p),abs(x2-x2p)+abs(y2-y2p))
    return 2*max(abs(x1-x1p)+abs(y1-y1p),abs(x2-x2p)+abs(y2-y2p))

    

a = np.zeros((5,5), dtype=np.int)

graph = wcd_utils.genGraph(a)
print(wcd_utils.wcd_astar(graph, ((2,0),(2,0)), ((0,4), (4,4)), h))
a[2][2] = 1
print(wcd_utils.wcd_astar(graph, ((2,0),(2,0)), ((0,4), (4,4)), h))
print(wcd_utils.wcd_astar(graph, ((2,0),(2,0)), ((0,4), (3,4)), h))

epsilon = 1/101

a = np.zeros((100,100), dtype=np.int)
a[50][50] = 1
graph = wcd_utils.genGraph(a)
print(wcd_utils.wcd_astar(graph, ((50,0),(50,0)), ((0,99), (99,99)), h))
