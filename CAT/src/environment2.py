from copy import deepcopy
import numpy as np
from enum import  IntEnum


class BlockWorld4Teams:
    ACTIONS = IntEnum('Actions', 'RIGHT LEFT NOOP QUERY PICKUP', start=0)
    def __init__(self, num_rooms, drop_room, num_blocks, locations, order, num_agents, agents, basecost=1,cost=0):
        self._num_rooms = num_rooms
        self._drop_room = drop_room
        self._num_blocks = num_blocks
        self._locations = np.array(locations)
        assert len(locations) == num_blocks
        self._order = np.array(order)
        self._current_block = 0
        self._num_agents = num_agents
        self.state = np.array([np.random.randint(num_rooms, size=num_agents), np.array(locations), np.array([-1 for _ in range(num_agents)]), np.array(self._order)])
        self._start_state = deepcopy(self.state)
        self._agents = agents
        self.basecost = basecost
        self.cost = cost

    def reset(self):
        self.state = deepcopy(self._start_state)
        self._current_block = 0
        for a in self._agents:
            a.reset(self.state)
        return self.state

    def step(self):
        temp_state = deepcopy(self.state)
        dropped = set()
        done = False
        reward = 0
        for i, a in enumerate(self._agents):
            a1, a2 = a(temp_state, i, self._drop_room)
            if a1 == BlockWorld4Teams.ACTIONS.RIGHT:
                self.state[0][i] += 1
                self.state[0][i] = min(self.state[0][i], self._num_rooms-1)
                if self.state[2][i] != -1:
                    self.state[1][self.state[2][i]] += 1
                    self.state[1][self.state[2][i]] = min(self.state[1][self.state[2][i]], self._num_rooms-1)
                reward += -1
            elif a1 == BlockWorld4Teams.ACTIONS.LEFT:
                self.state[0][i] -= 1
                self.state[0][i] = max(self.state[0][i], 0)
                if self.state[2][i] != -1:
                    self.state[1][self.state[2][i]] -= 1
                    self.state[1][self.state[2][i]] = max(self.state[1][self.state[2][i]], 0)
                reward += -1
            elif a1 == BlockWorld4Teams.ACTIONS.PICKUP:
                if self.state[1][a2] == self.state[0][i] and not any(k == a2 for k in self.state[2]):
                    self.state[2][i] = a2
                reward += -1
            elif a1 == BlockWorld4Teams.ACTIONS.QUERY:
                target, query = a2
                answer = self._agents[target].answer(self.state, query, target)
                a.receive(target, query, answer)
                reward += -1*(self.basecost + self.cost*len(query))
        for i,loc in enumerate(self.state[1]):
            if loc == -1:
                continue
            if loc == self._drop_room:
                self.state[1][i] = -1
                #print('i:',i)
                #print('loc:',loc)
                #print('state 2:', self.state[2])
                #print('state 2 == i:', np.array(self.state[2])==i)
                #ai = np.where(np.array(self.state[2]) == i)[0]
                ai = np.where(self.state[2] == i)[0]
                #print('ai:',ai)
                if len(ai) > 0:
                    self.state[2][ai] = -1
                dropped.add(i)
        while len(dropped) > 0:
            n = self._order[self._current_block]
            print('Dropped len: ',len(dropped))
            print('Dropped:', dropped)
            if n in dropped:
                print('n:',n)
                dropped.remove(n)
                #print('Dropped:', dropped)
                self._current_block += 1
                #reward += 1
            else:
                #print("done 1")
                done = True
                break
        if self._current_block == len(self._order):
            #print("done 2")
            done = True
        return self.state, reward, done, {}










