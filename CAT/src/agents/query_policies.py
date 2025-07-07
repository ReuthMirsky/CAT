from src.agents.agent import Policy
from src.environment import ToolFetchingEnvironment
from src.wcd_utils import fast_wcd
from src.utils import Point2D
import numpy as np
import random
from scipy.optimize import fsolve
from statistics import  median
import pulp
import copy
from skopt import gp_minimize
from skopt.space import Integer
from pyeasyga import pyeasyga
from src.agents.agent_utils import get_valid_actions



def is_ZB(obs, g1, g2):
    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
    valid_actions1 = np.array([True] * 4)
    if f_pos[0] <= t_pos[g1][0]:
        valid_actions1[1] = False # Left
    if f_pos[0] >= t_pos[g1][0]:
        valid_actions1[0] = False # Right
    if f_pos[1] >= t_pos[g1][1]:
        valid_actions1[2] = False # Down
    if f_pos[1] <= t_pos[g1][1]:
        valid_actions1[3] = False # Up

    valid_actions2 = np.array([True] * 4)
    if f_pos[0] <= t_pos[g2][0]: 
        valid_actions2[1] = False # Left
    if f_pos[0] >= t_pos[g2][0]:
        valid_actions2[0] = False # Right
    if f_pos[1] >= t_pos[g2][1]:
        valid_actions2[2] = False # Down
    if f_pos[1] <= t_pos[g2][1]:
        valid_actions2[3] = False # Up

    if not np.any(np.logical_and(valid_actions1, valid_actions2)):
        return True
    if t_pos[g1][0] == t_pos[g2][0] and t_pos[g1][1] == t_pos[g2][1] and f_pos[0] == t_pos[g1][0] and f_pos[1] == t_pos[g1][1]:
        return True
    return False



def never_query(obs, agent):
    return None


def random_query(obs, agent):
  #  if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
  #      return None
    goals = np.where(agent.probs > 0)[0]
    if np.all([agent.time < agent.wcd[g1,g2] for g1 in goals for g2 in goals if g1 != g2]):
        return None
    possible_stations = [s for s, s_p in enumerate(agent.probs) if s_p > 0]
    query =  random.sample(possible_stations, len(possible_stations) // 2)
    return query


def max_action_query(obs, agent):
    #if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
    #    return None
    goals = np.where(agent.probs > 0)[0]
    if np.all([agent.time < agent.wcd[g1,g2] for g1 in goals for g2 in goals if g1 != g2]):
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
    s_probs = agent.probs
    if np.all(t_pos[np.where(agent.probs > 0)] == f_pos):
        return random_query(obs, agent)

    stn_per_action = {
            ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.UP: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN: []
    }
    for i, t in enumerate(t_pos):
        if s_probs[i] == 0:
            continue
        if f_pos[0] < t[0]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT].append(i)
        if f_pos[0] > t[0]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT].append(i)
        if f_pos[1] < t[1]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.UP].append(i)
        if f_pos[1] > t[1]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN].append(i)

    query =  max(stn_per_action.values(), key=len)
    if len(query) == 0:
        return None
    if len(query) == len(goals):
        return None
    #print(query)
    return query

def min_action_query(obs, agent):
    #if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
    #    return None
    goals = np.where(agent.probs > 0)[0]
    if np.all([agent.time < agent.wcd[g1, g2] for g1 in goals for g2 in goals if g1 != g2]):
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
    s_probs = agent.probs

    if np.all(t_pos[np.where(agent.probs > 0)] == f_pos):
        return random_query(obs, agent)

    stn_per_action = {
            ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.UP: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN: []
    }
    for i, t in enumerate(t_pos):
        if s_probs[i] == 0:
            continue
        if f_pos[0] < t[0]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT].append(i)
        if f_pos[0] > t[0]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT].append(i)
        if f_pos[1] < t[1]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.UP].append(i)
        if f_pos[1] > t[1]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN].append(i)
    valid_queries = {k:v for k,v in stn_per_action.items() if len(v) > 0}

    return min(valid_queries.values(), key=len)

def median_action_query(obs, agent):
    #if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
    #    return None

    goals = np.where(agent.probs > 0)[0]
    if np.all([agent.time < agent.wcd[g1, g2] for g1 in goals for g2 in goals if g1 != g2]):
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
    s_probs = agent.probs
    if np.all(t_pos[np.where(agent.probs > 0)] == f_pos):
        return random_query(obs, agent)

    stn_per_action = {
            ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.UP: [],
            ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN: []
    }
    for i, t in enumerate(t_pos):
        if s_probs[i] == 0:
            continue
        if f_pos[0] < t[0]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT].append(i)
        if f_pos[0] > t[0]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT].append(i)
        if f_pos[1] < t[1]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.UP].append(i)
        if f_pos[1] > t[1]:
            stn_per_action[ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN].append(i)
    stns_per_action_values = list(stn_per_action.values())
    stns_per_action_values.sort(key=len)

    query = stns_per_action_values[len(stns_per_action_values)//2]
    if len(query) == 0:
        return None
    if len(query) == len(goals):
        return None
    return query

def smart_query(obs, agent):
    #if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
    #    return None
    goals = np.where(agent.probs > 0)[0]
    if np.all([agent.time < agent.wcd[g1, g2] for g1 in goals for g2 in goals if g1 != g2]):
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    zbs = set()
    goals = []
    zb_goals = set()
    for g in range(len(s_pos)):
        if agent.probs[g] == 0:
            continue
        goals.append(g)

    for g1 in goals:
        for g2 in goals:
            if g1 == g2:
                continue
            if (g2, g1) in zbs:
                continue
            if is_ZB(obs, g1, g2):
                zbs.add((g1, g2))
                zb_goals.add(g1)
                zb_goals.add(g2)

    bin1 = set()
    bin2 = set()
    used = set()

    problem = pulp.LpProblem("Find max split", pulp.LpMaximize)

    x = {i:pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Integer') for i in zb_goals}
    alpha = {(i,j):pulp.LpVariable(f"alpha_{i},{j}", lowBound=0, upBound=2, cat='Integer') for i,j in zbs}
    problem += pulp.lpSum(alpha[i,j] - x[i] - x[j] for i,j in zbs)
    for i,j in zbs:
        problem += alpha[i,j] <= 2*(x[i]+x[j])
    print("num pairs: ", len(zbs))
    print(len(problem.constraints))
    #problem.solve()
    problem.solve(pulp.apis.PULP_CBC_CMD(maxSeconds=120))

    for i in x:
        if x[i].varValue == 1:
            bin1.add(i)
        elif x[i].varValue == 0:
            bin2.add(i)
        else:
            raise ValueError

    remaining = []
    for g in goals:
        if g in zb_goals:
            continue
        remaining.append(g)
    query =  list(bin1) + random.sample(remaining, len(remaining)//2)
    if len(query) == 0:
        return None
    return query


def smart_query2(obs, agent):
    #if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
    #    return None
    goals = np.where(agent.probs > 0)[0]
    if np.all([agent.time < agent.wcd[g1, g2] for g1 in goals for g2 in goals if g1 != g2]):
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    zbs = set()
    goals = []
    zb_goals = set()
    for g in range(len(s_pos)):
        if agent.probs[g] == 0:
            continue
        goals.append(g)

    for g1 in goals:
        for g2 in goals:
            if g1 == g2:
                continue
            if (g2, g1) in zbs:
                continue
            if is_ZB(obs, g1, g2):
                zbs.add((g1, g2))
                zb_goals.add(g1)
                zb_goals.add(g2)

    bin1 = set()
    bin2 = set()
    used = set()

    problem = pulp.LpProblem("Find max split", pulp.LpMaximize)

    x = {i:pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Integer') for i in zb_goals}
    alpha = {(i,j):pulp.LpVariable(f"alpha_{i},{j}", lowBound=0, upBound=2, cat='Integer') for i,j in zbs}
    problem += pulp.lpSum((alpha[i,j] - x[i] - x[j]) * (agent.probs[i] + agent.probs[j]) for i,j in zbs)
    for i,j in zbs:
        problem += alpha[i,j] <= 2*(x[i]+x[j])
    #problem.solve()
    problem.solve(pulp.apis.PULP_CBC_CMD(maxSeconds=120))

    for i in x:
        if x[i].varValue == 1:
            bin1.add(i)
        elif x[i].varValue == 0:
            bin2.add(i)
        else:
            raise ValueError

    remaining = []
    for g in goals:
        if g in zb_goals:
            continue
        remaining.append(g)
    query = list(bin1) + random.sample(remaining, len(remaining)//2)
    if len(query) == 0:
        return None
    return query


def create_smart_query3(cost):
    def smart_query3(obs, agent):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        print(f_pos)
        print(t_pos)
        goals = np.where(agent.probs > 0)[0]
        if np.all([agent.time < agent.wcd[g1, g2] for g1 in goals for g2 in goals if g1 != g2]):
            return None
        #if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
            #if not np.all([agent.time < agent.wcd[g1, g2] for g1 in goals for g2 in goals if g1 != g2]):
                #input()
            #return None
    
    
        zbs = set()
        goals = []
        zb_goals = set()
        for g in range(len(s_pos)):
            if agent.probs[g] == 0:
                continue
            goals.append(g)
    
        for g1 in goals:
            for g2 in goals:
                if g1 == g2:
                    continue
                if (g2, g1) in zbs:
                    continue
                if is_ZB(obs, g1, g2):
                    zbs.add((g1, g2))
                    zb_goals.add(g1)
                    zb_goals.add(g2)
    
        bin1 = set()
        bin2 = set()
        used = set()
        print(zb_goals)
        print(zbs)
    
        problem = pulp.LpProblem("Find max split with cost", pulp.LpMaximize)
    
        x = {i:pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Integer') for i in zb_goals}
        alpha = {(i,j):pulp.LpVariable(f"alpha_{i},{j}", lowBound=0, upBound=2, cat='Integer') for i,j in zbs}
        problem += pulp.lpSum((alpha[i,j] - x[i] - x[j]) * (agent.probs[i] + agent.probs[j]) for i,j in zbs) - pulp.lpSum(cost * x[i] for i in x)
        for i,j in zbs:
            problem += alpha[i,j] <= 2*(x[i]+x[j])
        #problem.solve()
        problem.solve(pulp.apis.PULP_CBC_CMD(maxSeconds=120))
    
        for i in x:
            if x[i].varValue == 1:
                bin1.add(i)
            elif x[i].varValue == 0:
                bin2.add(i)
            else:
                raise ValueError
    
        remaining = []
        for g in goals:
            if g in zb_goals:
                continue
            remaining.append(g)
        query = list(bin1) + random.sample(remaining, len(remaining) // 2)
        print(query)
        if len(query) == 0:
            return None 
        return query
    return smart_query3

class interval:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def card(self):
        return self.b - self.a + 1

def get_voi(g, G, s1, s2, wcd_f, edp):
    intervals = [interval(wcd_f[gp,g][s2[0]][s2[1]], edp[g,gp][s1[0]][s1[1]]) for gp in G if g != gp and wcd_f[gp,g][s2[0]][s2[1]] <= edp[g,gp][s1[0]][s1[1]]]
    intervals.sort(key = lambda x:x.a)
    if len(intervals) == 0:
        return 0
    y = [ intervals[0] ]
    for i in intervals[1:]:
        if y[-1].b < i.a:
            y.append(i)
        elif y[-1].b >= i.a and i.b > y[-1].b:
            y[-1].b = i.b
    return np.sum([i.card() for i in y])


def create_optimal_query(cost, basecost, edp, wcd_f):

    def card(x):
        return x[1] - x[0]

        ##intervals = [interval(wcd_f[gp,g][s2[0]][s2[1]], edp[g,gp][s1[0]][s1[1]]) for gp in G if g != gp and wcd_f[gp,g][s2[0]][s2[1]] <= edp[g,gp][s1[0]][s1[1]]]
        #intervals = np.array([(wcd_f[gp,g][s2[0]][s2[1]]-1, edp[g,gp][s1[0]][s1[1]]) for gp in G if g != gp and wcd_f[gp,g][s2[0]][s2[1]] <= edp[g,gp][s1[0]][s1[1]]])
        #np.sort(intervals, axis=0)
        ##intervals.sort(key = lambda x:x.a)
        #if len(intervals) == 0:
        #    return 0
        #y = np.zeros(shape=(len(intervals), 2))
        #y[0] = intervals[0]
        #index = 1
        #for i in intervals[1:]:
        #    if y[index-1][1] < i[0]:
        #        y[index] = i
        #        index += 1
        #    elif y[index-1][1] >= i[0] and i[1] > y[index-1][1]:
        #        y[index-1][1] = i[1]
        #s =  np.sum(y, axis=0)
        #return s[1]-s[0]



    def sq(obs, agent):
        #if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
        #    return None
        goals = np.where(agent.probs > 0)[0]
        if np.all([agent.time < agent.wcd[g1, g2] for g1 in goals for g2 in goals if g1 != g2]):
            return None
    
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        G = np.where(agent.probs > 0)[0]
        probs = agent.probs[G]
        VOI = np.array([get_voi(g, G, w_pos, f_pos, wcd_f, edp) for g in G])
        values = {}
        def obj(x, data=None):
            x = np.array(x)
            #xt = x.tostring()
            xt = tuple(x)
            if xt in values:
                return values[xt]
            #G = np.where(agent.probs > 0)[0]
            #probs = agent.probs[G]

            G1 = G[np.where(x == 0)[0]]
            G2 = G[np.where(x == 1)[0]]

            #query = list(G1)



            w1 = agent.probs[G1]
            w2 = agent.probs[G2]

            VOI1 = np.array([get_voi(g, G1, w_pos, f_pos, wcd_f, edp) for g in G1])
            VOI2 = np.array([get_voi(g, G2, w_pos, f_pos, wcd_f, edp) for g in G2])

            value = -1*(np.dot(w1, VOI1) + np.dot(w2, VOI2) + cost*len(G1))
            values[xt] = value
            return value


        def crossover(x1, x2):
            i = np.random.randint(len(x1))
            new = np.empty(len(x1), dtype=int)
            new[:i] = x1[:i]
            new[i:] = x2[i:]
            new2 = np.empty(len(x1), dtype=int)
            new2[:i] = x2[:i]
            new2[i:] = x1[i:]
            return new, new2

        def mutation(x, prob=0.001):
            return x^np.random.choice([0,1], size=len(x), replace=True, p=[1-prob, prob])

        def select(pop, fitness):
            m = np.random.randint(50, size=2)
            if fitness[m[0]] >= fitness[m[1]]:
                return pop[m[0]]
            return pop[m[1]]


        population = np.random.randint(2, size=(50, len(G)))
        fitness = np.array([obj(x) for x in population])
        best = population[np.argmax(fitness)]
        bestFit = fitness[best]
        for _ in range(100):
            new_pop = np.empty(population.shape, dtype=int)
            m = 0
            while m < len(new_pop):
                p1 = select(population, fitness)
                p2 = select(population, fitness)
                new,new2 = crossover(p1, p2)
                new = mutation(new)
                new2 = mutation(new2)
                new_pop[m] = new
                m += 1
                if m < len(new_pop):
                    new_pop[m] = new2
                    m += 1
            population = new_pop
            fitness = np.array([obj(x) for x in population])
            if np.all(np.max(fitness) > bestFit):
                best = population[np.argmax(fitness)]
                bestFit = fitness[best]



        answer = (bestFit, best)
        print("answer",answer)
        objective = obj(answer[1], None)
        objective += (np.dot(VOI, agent.probs[G]) - basecost)
        query = list(G[np.where(np.array(answer[1]) == 0)[0]])
        print(query, objective)
        if objective > 0:
            if len(query) == 0 or len(query) == len(G):
                print("ERROR: Not finding optimal")
                return None
            print("Asking Query: ", query)
            return query
        return None

    return sq


def create_optimal_query2(cost_fun, edp, wcd_f):

    def card(x):
        return x[1] - x[0]

        ##intervals = [interval(wcd_f[gp,g][s2[0]][s2[1]], edp[g,gp][s1[0]][s1[1]]) for gp in G if g != gp and wcd_f[gp,g][s2[0]][s2[1]] <= edp[g,gp][s1[0]][s1[1]]]
        #intervals = np.array([(wcd_f[gp,g][s2[0]][s2[1]]-1, edp[g,gp][s1[0]][s1[1]]) for gp in G if g != gp and wcd_f[gp,g][s2[0]][s2[1]] <= edp[g,gp][s1[0]][s1[1]]])
        #np.sort(intervals, axis=0)
        ##intervals.sort(key = lambda x:x.a)
        #if len(intervals) == 0:
        #    return 0
        #y = np.zeros(shape=(len(intervals), 2))
        #y[0] = intervals[0]
        #index = 1
        #for i in intervals[1:]:
        #    if y[index-1][1] < i[0]:
        #        y[index] = i
        #        index += 1
        #    elif y[index-1][1] >= i[0] and i[1] > y[index-1][1]:
        #        y[index-1][1] = i[1]
        #s =  np.sum(y, axis=0)
        #return s[1]-s[0]



    def sq(obs, agent):
        #if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
        #    return None
        goals = np.where(agent.probs > 0)[0]
        if np.all([agent.time < agent.wcd[g1, g2] for g1 in goals for g2 in goals if g1 != g2]):
            return None
    
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        G = np.where(agent.probs > 0)[0]
        probs = agent.probs[G]
        VOI = np.array([get_voi(g, G, w_pos, f_pos, wcd_f, edp) for g in G])
        values = {}
        def obj(x, data=None):
            x = np.array(x)
            #xt = x.tostring()
            xt = tuple(x)
            if xt in values:
                return values[xt]
            #G = np.where(agent.probs > 0)[0]
            #probs = agent.probs[G]

            G1 = G[np.where(x == 0)[0]]
            G2 = G[np.where(x == 1)[0]]

            #query = list(G1)



            w1 = agent.probs[G1]
            w2 = agent.probs[G2]

            VOI1 = np.array([get_voi(g, G1, w_pos, f_pos, wcd_f, edp) for g in G1])
            VOI2 = np.array([get_voi(g, G2, w_pos, f_pos, wcd_f, edp) for g in G2])

            value = -1*(np.dot(w1, VOI1) + np.dot(w2, VOI2) + cost_fun(obs, x))
            values[xt] = value
            return value


        def crossover(x1, x2):
            i = np.random.randint(len(x1))
            new = np.empty(len(x1), dtype=int)
            new[:i] = x1[:i]
            new[i:] = x2[i:]
            new2 = np.empty(len(x1), dtype=int)
            new2[:i] = x2[:i]
            new2[i:] = x1[i:]
            return new, new2

        def mutation(x, prob=0.001):
            return x^np.random.choice([0,1], size=len(x), replace=True, p=[1-prob, prob])

        def select(pop, fitness):
            m = np.random.randint(50, size=2)
            if fitness[m[0]] >= fitness[m[1]]:
                return pop[m[0]]
            return pop[m[1]]


        population = np.random.randint(2, size=(50, len(G)))
        fitness = np.array([obj(x) for x in population])
        best = population[np.argmax(fitness)]
        bestFit = fitness[best]
        for _ in range(100):
            new_pop = np.empty(population.shape, dtype=int)
            m = 0
            while m < len(new_pop):
                p1 = select(population, fitness)
                p2 = select(population, fitness)
                new,new2 = crossover(p1, p2)
                new = mutation(new)
                new2 = mutation(new2)
                new_pop[m] = new
                m += 1
                if m < len(new_pop):
                    new_pop[m] = new2
                    m += 1
            population = new_pop
            fitness = np.array([obj(x) for x in population])
            if np.all(np.max(fitness) > bestFit):
                best = population[np.argmax(fitness)]
                bestFit = fitness[best]



        answer = (bestFit, best)
        print("answer",answer)
        objective = obj(answer[1], None)
        objective += (np.dot(VOI, agent.probs[G]))
        query = list(G[np.where(np.array(answer[1]) == 0)[0]])
        print(query, objective)
        if objective > 0:
            if len(query) == 0 or len(query) == len(G):
                print("ERROR: Not finding optimal")
                return None
            print("Asking Query: ", query)
            return query
        return None

    return sq




def smart_query_noRandom(obs, agent):
    #if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
    #    return None
    goals = np.where(agent.probs > 0)[0]
    if np.all([agent.time < agent.wcd[g1, g2] for g1 in goals for g2 in goals if g1 != g2]):
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    zbs = set()
    goals = []
    zb_goals = set()
    for g in range(len(s_pos)):
        if agent.probs[g] == 0:
            continue
        goals.append(g)

    for g1 in goals:
        for g2 in goals:
            if g1 == g2:
                continue
            if (g2, g1) in zbs:
                continue
            if is_ZB(obs, g1, g2):
                zbs.add((g1, g2))
                zb_goals.add(g1)
                zb_goals.add(g2)

    bin1 = set()
    bin2 = set()
    used = set()

    problem = pulp.LpProblem("Find max split", pulp.LpMaximize)

    x = {i:pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Integer') for i in zb_goals}
    alpha = {(i,j):pulp.LpVariable(f"alpha_{i},{j}", lowBound=0, upBound=2, cat='Integer') for i,j in zbs}
    problem += pulp.lpSum(alpha[i,j] - x[i] - x[j] for i,j in zbs)
    for i,j in zbs:
        problem += alpha[i,j] <= 2*(x[i]+x[j])
    #problem.solve()
    problem.solve(pulp.apis.PULP_CBC_CMD(maxSeconds=120))

    for i in x:
        if x[i].varValue == 1:
            bin1.add(i)
        elif x[i].varValue == 0:
            bin2.add(i)
        else:
            raise ValueError

    remaining = []
    for g in goals:
        if g in zb_goals:
            continue
        remaining.append(g)
    #return list(bin1) + random.sample(remaining, len(remaining)//2)
    if len(bin1) == 0:
        print(zbs)
        print(f_pos)
        print(t_pos)
        print(goals)
        for g in goals:
            print(t_pos[g])
        return None
    return list(bin1)


def smart_query2_noRandom(obs, agent):
    #if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
    #    return None
    goals = np.where(agent.probs > 0)[0]
    if np.all([agent.time < agent.wcd[g1, g2] for g1 in goals for g2 in goals if g1 != g2]):
        return None

    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    zbs = set()
    goals = []
    zb_goals = set()
    for g in range(len(s_pos)):
        if agent.probs[g] == 0:
            continue
        goals.append(g)

    for g1 in goals:
        for g2 in goals:
            if g1 == g2:
                continue
            if (g2, g1) in zbs:
                continue
            if is_ZB(obs, g1, g2):
                zbs.add((g1, g2))
                zb_goals.add(g1)
                zb_goals.add(g2)

    bin1 = set()
    bin2 = set()
    used = set()

    problem = pulp.LpProblem("Find max split", pulp.LpMaximize)

    x = {i:pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Integer') for i in zb_goals}
    alpha = {(i,j):pulp.LpVariable(f"alpha_{i},{j}", lowBound=0, upBound=2, cat='Integer') for i,j in zbs}
    problem += pulp.lpSum((alpha[i,j] - x[i] - x[j]) * (agent.probs[i] + agent.probs[j]) for i,j in zbs)
    for i,j in zbs:
        problem += alpha[i,j] <= 2*(x[i]+x[j])
    #problem.solve()
    problem.solve(pulp.apis.PULP_CBC_CMD(maxSeconds=120))

    for i in x:
        if x[i].varValue == 1:
            bin1.add(i)
        elif x[i].varValue == 0:
            bin2.add(i)
        else:
            raise ValueError

    remaining = []
    for g in goals:
        if g in zb_goals:
            continue
        remaining.append(g)
    #return list(bin1) + random.sample(remaining, len(remaining)//2)
    if len(bin1) == 0:
        #print("zb_goals len", len(zb_goals))
        #print("zbs len", len(zbs))
        #for i in zb_goals:
            #print(agent.probs[i])
        return None

    return list(bin1)


def create_smart_query3_noRandom(cost):
    def smart_query3_noRandom(obs, agent):
        #if np.any(get_valid_actions(obs, agent)) or np.max(agent.probs) >= 1:
        #    return None
        goals = np.where(agent.probs > 0)[0]
        if np.all([agent.time < agent.wcd[g1, g2] for g1 in goals for g2 in goals if g1 != g2]):
            return None
    
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
    
        zbs = set()
        goals = []
        zb_goals = set()
        for g in range(len(s_pos)):
            if agent.probs[g] == 0:
                continue
            goals.append(g)
    
        for g1 in goals:
            for g2 in goals:
                if g1 == g2:
                    continue
                if (g2, g1) in zbs:
                    continue
                if is_ZB(obs, g1, g2):
                    zbs.add((g1, g2))
                    zb_goals.add(g1)
                    zb_goals.add(g2)
    
        bin1 = set()
        bin2 = set()
        used = set()
    
        problem = pulp.LpProblem("Find max split with cost", pulp.LpMaximize)
    
        x = {i:pulp.LpVariable(f"x_{i}", lowBound=0, upBound=1, cat='Integer') for i in zb_goals}
        alpha = {(i,j):pulp.LpVariable(f"alpha_{i},{j}", lowBound=0, upBound=2, cat='Integer') for i,j in zbs}
        problem += pulp.lpSum((alpha[i,j] - x[i] - x[j]) * (agent.probs[i] + agent.probs[j]) for i,j in zbs) - pulp.lpSum(cost * x[i] for i in x)
        for i,j in zbs:
            problem += alpha[i,j] <= 2*(x[i]+x[j])
        #problem.solve()
        problem.solve(pulp.apis.PULP_CBC_CMD(maxSeconds=120))
    
        for i in x:
            if x[i].varValue == 1:
                bin1.add(i)
            elif x[i].varValue == 0:
                bin2.add(i)
            else:
                raise ValueError
    
        remaining = []
        for g in goals:
            if g in zb_goals:
                continue
            remaining.append(g)
        #return list(bin1) + random.sample(remaining, len(remaining)//2)
        if len(bin1) == 0:
            return None
        return list(bin1)
    return smart_query3_noRandom
