import copy
import numpy as np
import random
from src.environment import ToolFetchingEnvironment
#from environment import ToolFetchingEnvironment



move_actions = [ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT, ToolFetchingEnvironment.WORKER_ACTIONS.LEFT, ToolFetchingEnvironment.WORKER_ACTIONS.UP, ToolFetchingEnvironment.WORKER_ACTIONS.DOWN, ToolFetchingEnvironment.WORKER_ACTIONS.WORK]

def move_transition(i, j, a, width=10, height=10):
    if a == ToolFetchingEnvironment.WORKER_ACTIONS.WORK:
        return i,j
    if a == ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT:
        i += 1
    if a == ToolFetchingEnvironment.WORKER_ACTIONS.LEFT:
        i -= 1
    if a == ToolFetchingEnvironment.WORKER_ACTIONS.UP:
        j += 1
    if a == ToolFetchingEnvironment.WORKER_ACTIONS.DOWN:
        j -= 1
    i = min(i, width-1)
    i = max(i, 0)
    j = min(j, height-1)
    j = max(j, 0)
    return i, j


def num_plans(i, j, k, l):
    dx = abs(i-k)
    dy = abs(j-l)
    return np.math.factorial(dx+dy)/(np.math.factorial(dx)*np.math.factorial(dy))

def random_optimal_plan(i, j, g, a):
    probs = np.zeros(4)
    if i == g[0] and j == g[1]:
        if a == ToolFetchingEnvironment.WORKER_ACTIONS.WORK:
            return 1
        else:
            return 0
    else:
        if a == ToolFetchingEnvironment.WORKER_ACTIONS.WORK:
            return 0
        if i < g[0]:
            probs[ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT] = num_plans(i+1, j, g[0], g[1])
        if i > g[0]:
            probs[ToolFetchingEnvironment.WORKER_ACTIONS.LEFT] = num_plans(i-1, j, g[0], g[1])
        if j < g[1]:
            probs[ToolFetchingEnvironment.WORKER_ACTIONS.UP] = num_plans(i, j+1, g[0], g[1])
        if j > g[1]:
            probs[ToolFetchingEnvironment.WORKER_ACTIONS.DOWN] = num_plans(i, j-1, g[0], g[1])
        probs /= np.sum(probs)
        #assert np.sum(probs) == 1, probs
        return probs[a]



def random_optimal(i, j, g, a):
    probs = np.zeros(4) 
    if i == g[0] and j == g[1]:
        if a == ToolFetchingEnvironment.WORKER_ACTIONS.WORK:
            return 1
        else:
            return 0
    else:
        if a == ToolFetchingEnvironment.WORKER_ACTIONS.WORK:
            return 0
    if i < g[0]:
        probs[ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT] = 1
    if i > g[0]:
        probs[ToolFetchingEnvironment.WORKER_ACTIONS.LEFT] = 1
    if j > g[1]:
        probs[ToolFetchingEnvironment.WORKER_ACTIONS.DOWN] = 1
    if j < g[1]:
        probs[ToolFetchingEnvironment.WORKER_ACTIONS.UP] = 1
    probs /= np.sum(probs)
    return probs[a]



def ACD_iter(g1, g2, pi, T, actions, width=10, height=10, epsilon=0.1):
    V = np.zeros((width, height))
    diff = float('inf')
    while diff > epsilon:
        diff = 0
        Vp = np.zeros((width, height))
        for i in range(width):
            for j in range(height):
                same_prob = 0
                for a in actions:
                    Vp[i,j] += pi(i, j, g1, a)*pi(i, j, g2, a)*(1 + V[T(i, j, a, width, height)])
                    same_prob += pi(i,j, g1, a)*pi(i, j, g2, a)
                Vp[i,j] += (1-same_prob)
                diff = max(diff, abs(Vp[i, j] - V[i, j]))
        V = Vp
    return V

def ACD(G, pi=random_optimal, T=move_transition, actions=move_actions, width=10, height=10, epsilon=0.01):
    acd = {}
    for i,g1 in enumerate(G):
        for j,g2 in enumerate(G):
            if i == j: continue
            #print(i,j)
            acd[i,j] = ACD_iter(g1, g2, pi, T, actions, width, height, epsilon)
    return acd

def ACD_iter2(g1, g2, pi, T, actions, width=10, height=10, epsilon=0.01):
    V = np.zeros((width, height))
    diff = float('inf')
    while diff > epsilon:
        print(diff)
        diff = 0
        Vp = np.zeros((width, height))
        for i in range(width):
            for j in range(height):
                same_prob = 0
                for a in actions:
                    Vp[i,j] += pi(i, j, g1, a)*np.ceil(pi(i, j, g2, a))*(1 + V[T(i, j, a, width, height)])
                    #Vp[i,j] += pi(i, j, g1, a)*(1-np.ceil(pi(i, j, g2, a)))
                    same_prob += pi(i,j, g1, a)*np.ceil(pi(i, j, g2, a))
                Vp[i,j] += (1-same_prob)
                diff = max(diff, abs(Vp[i, j] - V[i, j]))
        print(V, Vp)
        V = Vp
    return V

def ACD2(G, pi=random_optimal_plan, T=move_transition, actions=move_actions, width=10, height=10, epsilon=0.01):
    acd = {}
    for i,g1 in enumerate(G):
        for j,g2 in enumerate(G):
            if i == j: continue
            print(i,j)
            acd[i,j] = ACD_iter2(g1, g2, pi, T, actions, width, height, epsilon)
    return acd

def WCD_iter(g1, g2, pi, T, actions, width=10, height=10, epsilon=0.01):
    V = np.zeros((width, height))
    diff = float('inf')
    if np.all(g1 == g2):
        for i in range(width):
            for j in range(height):
                V[i,j] = abs(i-g1[0]) + abs(j-g2[1]) + 1
        return V
    while diff > epsilon:
        diff = 0
        Vp = np.zeros((width, height))
        for i in range(width):
            for j in range(height):
                same_prob = 0
                worst = 0
                for a in actions:
                    worst = np.max([worst, np.ceil(pi(i, j, g1, a))*np.ceil(pi(i, j, g2, a))*(1 + V[T(i, j, a, width, height)])])
                Vp[i,j] = max(worst, 1)
                diff = max(diff, abs(Vp[i, j] - V[i, j]))
        V = Vp
    return V

def WCD(G, pi=random_optimal, T=move_transition, actions=move_actions, width=10, height=10, epsilon=0.01):
    wcd = {}
    for i,g1 in enumerate(G):
        for j,g2 in enumerate(G):
            if i == j: continue
            print(i,j)
            wcd[i,j] = WCD_iter(g1, g2, pi, T, actions, width, height, epsilon)
    return wcd


if __name__ == '__main__':
    #pos = [(i,j) for i in range(50) for j in range(50)]
    #goals = random.sample(pos, 20)
    goals = [np.array([7,2]), np.array([7,8]), np.array([3, 8])]
    wcd = ACD2(goals, width=10, height=10, pi=random_optimal_plan)
    print((wcd[1,2] + wcd[2,1])/2)
    np.savetxt("edp_g1_g2.csv", wcd[1,2])
    np.savetxt("edp_g2_g1.csv", wcd[2,1])
    #np.savetxt("test_acd.csv", (wcd[1,2] + wcd[2,1])/2)
