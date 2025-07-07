from src import wcd_utils as wcd
import numpy as np

def get_ZQ(goals, grid, toolbox, worker, fetcher):
    ZQ = {}
    graph = wcd.genGraph(grid)
    epsilon = 1/(len(grid)*len(grid[0])+1)
    def h(s1, s2):
        p1,p2 = s1
        p1p,p2p = s2
        if p1 == p2:
            return (2-epsilon)*max(abs(p1[0]-p1p[0]) + abs(p1[1]-p1p[1]), abs(p2[0]-p2p[0]) + abs(p2[1]-p2p[1]))
        else:
            return (2)*max(abs(p1[0]-p1p[0]) + abs(p1[1]-p1p[1]), abs(p2[0]-p2p[0]) + abs(p2[1]-p2p[1]))
    ZP = abs(worker[0]-toolbox[0]) + abs(worker[1]-toolbox[1])
    for g1 in goals:
        for g2 in goals:
            if g1 == g2: continue
            plan = wcd.wcd_astar(graph, (worker,worker), (g1,g2), h)
            for i,p in enumerate(plan):
                p1,p2 = p
                if p1 != p2:
                    if i > ZP:
                        ZQ[(g1,g2)] = (ZP, i)
                    break
    return ZQ

