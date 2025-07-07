import numpy as np
from copy import deepcopy
from src.environment2 import BlockWorld4Teams

def create_optimal_query(cost, basecost, edp, wcd_f):
    class interval:
        def __init__(self, a, b):
            self.a = a
            self.b = b

        def card(self):
            return self.b-self.a

    def get_voi(g, G, s1, s2):
        intervals = [interval(wcd_f[gp,g][s2], edp[g,gp][s1]) for gp in G if g != gp and wcd_f[gp,g][s2] <= edp[g,gp][s1]]
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



    def sq(obs, agent, a_index, drop_room):
        a_pos, b_pos, ab, order = obs
        cur_block = len(np.where(b_pos == -1)[0])
        goals = order[cur_block:cur_block+len(a_pos)]
        target = -1
        for t, probs in enumerate(agent._probs):
            if a_index == t:
                continue
            probs2 = probs[goals]
            probs2 /= np.sum(probs2)
            if (probs2 < 1-agent._epsilon).all():
                target = t
                break

        if target == -1:
            return None

    
        #G = np.where(agent._probs[target] > 0)[0]
        G = goals
        probs = agent._probs[target][G]
        probs /= np.sum(probs)
        VOI = np.array([get_voi(g, G, a_pos[target], a_pos[a_index]) for g in G])
        values = {}


        def obj(x, data=None):
            x = np.array(x)
            xt = tuple(x)
            if xt in values:
                return values[xt]
            #G = np.where(agent._probs[target] > 0)[0]
            probs = agent._probs[target][G]
            probs /= np.sum(probs)

            G1 = G[np.where(x == 0)[0]]
            G2 = G[np.where(x == 1)[0]]

            query = list(G1)


            

            w1 = agent._probs[target][G1]
            w1 /= np.sum(w1)
            w2 = agent._probs[target][G2]
            w2 /= np.sum(w2)

            VOI1 = np.array([get_voi(g, G1, a_pos[target], a_pos[a_index]) for g in G1])
            VOI2 = np.array([get_voi(g, G2, a_pos[target], a_pos[a_index]) for g in G2])

            value =  -1*(np.dot(w1, VOI1) + np.dot(w2, VOI2) + cost*len(G1))
            values[xt] = value
            #if len(G1) == 0:
            #    print("Empty Query")
            #    print(x)
            #    print(G, G2)
            #    print(value, np.dot(VOI, agent.probs[G]) + value - basecost)
            #print(query, np.dot(VOI, agent.probs[G]) + value - basecost)
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
            m = np.random.randint(2, size=2)
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



        """

        ga  = pyeasyga.GeneticAlgorithm(G)
        ga.population_size = 200
        ga.fitness_function = obj
        ga.run()

        space = [Integer(0, 1) for _ in G]
        #answer = gp_minimize(obj, space)
        answer = ga.best_individual()
        """
        answer = (bestFit, best)
        print("answer",answer)
        objective = obj(answer[1], None)
        #objective += (np.dot(VOI, agent._probs[target][G]) - basecost)
        objective += (np.dot(VOI, probs) - basecost)
        query = list(G[np.where(np.array(answer[1]) == 0)[0]])
        print(query, objective)
        if objective > 0:
            if len(query) == 0 or len(query) == len(G):
                print("ERROR: Not finding optimal")
                return None
            print("Asking Query: ", query)
            return target, query
        return None

    return sq


class Agent:
    def answer(self, state, query, a_index):
        raise NotImplementedError

    def __call__(self, state, a_index, drop_room):
        raise NotImplementedError

    def receive(self, target, query, answer):
        raise NotImplementedError

    def reset(self, state):
        raise NotImplementedError


def action_to_goal(pos, goal):
    if pos < goal:
        return BlockWorld4Teams.ACTIONS.RIGHT, None
    elif pos > goal:
        return BlockWorld4Teams.ACTIONS.LEFT, None
    else:
        return BlockWorld4Teams.ACTIONS.NOOP, None


class GreedyAgent(Agent):
    def __init__(self):
        pass

    def __call__(self, state, a_index, drop_room):
        a_pos, b_pos, ab, order = state
        current_block = len(np.where(b_pos == -1)[0])
        drop_block = current_block
        while order[current_block] in ab:
            if np.where(ab == order[current_block])[0] == a_index:
                break
            current_block += 1
            if current_block >= len(order):
                break
        if current_block >= len(order):
            return BlockWorld4Teams.ACTIONS.NOOP, None
        target = order[current_block]
        loc = a_pos[a_index]
        t_loc = b_pos[target]
        if target == ab[a_index]:
            if current_block == drop_block:
                return action_to_goal(loc, drop_room)
            else:
                if abs(drop_room - loc) > 1:
                    return action_to_goal(loc, drop_room)
                else:
                    return BlockWorld4Teams.ACTIONS.NOOP, None
        else:
            if loc == t_loc:
                return BlockWorld4Teams.ACTIONS.PICKUP, target
            return action_to_goal(loc, t_loc)

    def receive(self, query, answer):
        pass

    def reset(self, state):
        pass

    def answer(self, state, query, a_index):
        a_pos, b_pos, ab, order = state
        current_block = len(np.where(b_pos == -1)[0])
        while order[current_block] in ab:
            if np.where(ab == order[current_block])[0] == a_index:
                break
            current_block += 1
        target = order[current_block]
        return target in query

        
def never_query(obs, agent, a_index, drop_room):
    return None

class SmartAgent(Agent):
    def __init__(self, probs, query_policy=never_query, epsilon=0.01):
        self._probs = deepcopy(probs)
        self._start_probs = deepcopy(probs)
        self.last_obs = None
        self.query_policy = query_policy
        self._epsilon = epsilon

    def make_inference(self, obs, a_index, drop_room):
        if self.last_obs is None:
            self.last_obs = deepcopy(obs)
            return

        a_pos, b_pos, ab, order = obs
        la_pos, lb_pos, lab, lorder = self.last_obs

        for i, p in enumerate(a_pos):
            if i == a_index:
                continue
            if p == drop_room:
                self._probs[i] = deepcopy(self._start_probs[i])
                self._probs[i][np.where(b_pos == -1)[0]] = 0
                self._probs[i] /= np.sum(self._probs[i])
                #print("case 1")
                continue
            if ab[i] != -1:
                for j in range(len(self._probs[i])):
                    if j == ab[i]:
                        self._probs[i][j] = 1
                    else:
                        self._probs[i][j] = 0
                #print("case 2")
                continue
            #print("Deets")
            #print(a_pos)
            #print(i)
            #print(p)
            #print(la_pos)
            #print("Deets done")
            delta = p-la_pos[i]
            if delta > 0:
                for j, b in enumerate(b_pos):
                    if b < p:
                        self._probs[i][j] *= self._epsilon
            elif delta < 0:
                for j, b in enumerate(b_pos):
                    if b > p:
                        self._probs[i][j] *= self._epsilon
            else:
                for j, b in enumerate(b_pos):
                    if b != p:
                        self._probs[i][j] *= self._epsilon
            self._probs[i] /= np.sum(self._probs[i])
            #print("delta:",delta)
            #print("case 3")

        self.last_obs = deepcopy(obs)

    def __call__(self, state, a_index, drop_room):
        self.make_inference(state, a_index, drop_room)
        a_pos, b_pos, ab, order = state
        loc = a_pos[a_index]

        current_block = len(np.where(b_pos == -1)[0])
        drop_block = current_block
        if ab[a_index] != -1:
            if ab[a_index] == order[drop_block]:
                return action_to_goal(loc, drop_room)
            else:
                if abs(drop_room - loc) > 1:
                    return action_to_goal(loc, drop_room)
                else:
                    return BlockWorld4Teams.ACTIONS.NOOP, None
        valid_blocks = order[current_block:current_block+len(a_pos)]
        target_blocks = []
        for i,v in enumerate(valid_blocks):
            if (self._probs[:, v] >= 1-self._epsilon).any():
                continue
            target_blocks.append(v)
        target_blocks = np.array(target_blocks)
        if len(target_blocks) == 0:
            return BlockWorld4Teams.ACTIONS.NOOP, None

        if (b_pos[target_blocks] > loc).all():
            return BlockWorld4Teams.ACTIONS.RIGHT, None
        elif (b_pos[target_blocks] < loc).all():
            return BlockWorld4Teams.ACTIONS.LEFT, None

        query = self.query_policy(state, self, a_index, drop_room)
        if query is not None:
            return BlockWorld4Teams.ACTIONS.QUERY, query
        
        if len(target_blocks) == 1:
            target = target_blocks[0]
            t_loc = b_pos[target]
            if target == ab[a_index]:
                if current_block == drop_block:
                    return action_to_goal(loc, drop_room)
                else:
                    if abs(drop_room - loc) > 1:
                        return action_to_goal(loc, drop_room)
                    else:
                        return BlockWorld4Teams.ACTIONS.NOOP, None
            else:
                if loc == t_loc:
                    return BlockWorld4Teams.ACTIONS.PICKUP, target
                return action_to_goal(loc, t_loc)
        return BlockWorld4Teams.ACTIONS.NOOP, None

    def receive(self, target, query, answer):
        if answer:
            for j in range(len(self._probs[target])):
                if j not in query:
                    self._probs[target][j] *= self._epsilon
        else:
            for j in range(len(self._probs[target])):
                if j in query:
                    self._probs[target][j] *= self._epsilon
        self._probs[target] /= np.sum(self._probs[target])


    def reset(self, state):
        self._probs = deepcopy(self._start_probs)

    def answer(self, state, query, a_index):
        a_pos, b_pos, ab, order = state
        current_block = len(np.where(b_pos == -1)[0])
        while order[current_block] in ab:
            if np.where(ab == order[current_block])[0] == a_index:
                break
            current_block += 1
        target = order[current_block]
        return target in query

