import copy
import numpy as np
import random
from src.environment2 import BlockWorld4Teams
#from environment2 import BlockWorld4Teams
#from environment import ToolFetchingEnvironment


move_actions = [BlockWorld4Teams.ACTIONS.LEFT, BlockWorld4Teams.ACTIONS.RIGHT, BlockWorld4Teams.ACTIONS.PICKUP]



def move_transition(i, a, width=10):
    if a == BlockWorld4Teams.ACTIONS.PICKUP:
        return i
    if a == BlockWorld4Teams.ACTIONS.RIGHT:
        i += 1
    if a == BlockWorld4Teams.ACTIONS.LEFT:
        i -= 1
    i = min(i, width-1)
    i = max(i, 0)
    return i


def num_plans(i, j, k, l):
    dx = abs(i-k)
    dy = abs(j-l)
    return np.math.factorial(dx+dy)/(np.math.factorial(dx)*np.math.factorial(dy))

def random_optimal_plan(i, g, a):
    probs = np.zeros(2)
    if i == g:
        if a == BlockWorld4Teams.ACTIONS.PICKUP:
            return 1
        else:
            return 0
    else:
        if a == BlockWorld4Teams.ACTIONS.PICKUP:
            return 0
        if i < g:
            probs[BlockWorld4Teams.ACTIONS.RIGHT] = 1
        if i > g:
            probs[BlockWorld4Teams.ACTIONS.LEFT] = 1
        probs /= np.sum(probs)
        #assert np.sum(probs) == 1, probs
        return probs[a]



def ACD_iter2(g1, g2, pi, T, actions, width=10, epsilon=0.01):
    V = np.zeros(width)
    if g1 == g2:
        for i in range(width):
            V[i] = abs(i-g1)+1
        return V
    diff = float('inf')
    while diff > epsilon:
        diff = 0
        Vp = np.zeros(width)
        for i in range(width):
            same_prob = 0
            for a in actions:
                Vp[i] += pi(i, g1, a)*np.ceil(pi(i, g2, a))*(1 + V[T(i, a, width)])
                #Vp[i,j] += pi(i, j, g1, a)*(1-np.ceil(pi(i, j, g2, a)))
                same_prob += pi(i, g1, a)*np.ceil(pi(i, g2, a))
            Vp[i] += (1-same_prob)
            #print(diff)
            diff = max(diff, abs(Vp[i] - V[i]))
        V = Vp
    return V

def ACD2(G, pi=random_optimal_plan, T=move_transition, actions=move_actions, width=10, epsilon=0.01):
    acd = {}
    for i,g1 in enumerate(G):
        for j,g2 in enumerate(G):
            if i == j: continue
            print(i,j)
            acd[i,j] = ACD_iter2(g1, g2, pi, T, actions, width, epsilon)
    return acd

def WCD_iter(g1, g2, pi, T, actions, width=10, epsilon=0.01):
    V = np.zeros(width)
    if g1 == g2:
        for i in range(width):
            V[i] = abs(i-g1)+1
        return V
    diff = float('inf')
    if np.all(g1 == g2):
        for i in range(width):
            V[i] = abs(i-g1)
        return V
    while diff > epsilon:
        diff = 0
        Vp = np.zeros(width)
        for i in range(width):
            same_prob = 0
            worst = 0
            for a in actions:
                worst = np.max([worst, np.ceil(pi(i, g1, a))*np.ceil(pi(i, g2, a))*(1 + V[T(i, a, width)])])
            Vp[i] = max(worst, 1)
            diff = max(diff, abs(Vp[i] - V[i]))
        V = Vp
    return V

def WCD(G, pi=random_optimal_plan, T=move_transition, actions=move_actions, width=10, epsilon=0.01):
    wcd = {}
    for i,g1 in enumerate(G):
        for j,g2 in enumerate(G):
            if i == j: continue
            print(i,j)
            wcd[i,j] = WCD_iter(g1, g2, pi, T, actions, width, epsilon)
    return wcd


if __name__ == '__main__':
    #pos = [(i,j) for i in range(50) for j in range(50)]
    #goals = random.sample(pos, 20)
    goals = [7, 5 ,3]
    wcd = ACD2(goals, width=10, pi=random_optimal_plan)
    print(wcd)
    wcd = WCD(goals, width=10, pi=random_optimal_plan)
    print(wcd)
    #print((wcd[1,2] + wcd[2,1])/2)
    #np.savetxt("edp_g1_g2.csv", wcd[1,2])
    #np.savetxt("edp_g2_g1.csv", wcd[2,1])
    #np.savetxt("test_acd.csv", (wcd[1,2] + wcd[2,1])/2)
