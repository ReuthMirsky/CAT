import numpy as np
from src.environment import ToolFetchingEnvironment

def actions_to_goal(pos, goal):
    actions = []
    if pos[0] < goal[0]:
        actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT)
    elif pos[0] > goal[0]:
        actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.LEFT)
    if pos[1] > goal[1]:
        actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.DOWN)
    elif pos[1] < goal[1]:
        actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.UP)
    if len(actions) == 0:
        return set([ToolFetchingEnvironment.WORKER_ACTIONS.WORK])
    return set(actions)

class Model:
    def init(self, obs):
        raise NotImplementedError

    def __call__(self, obs, goal):
        raise NotImplementedError

class EpsilonGreedyModel(Model):
    def __init__(self, epsilon):
        self._epsilon = epsilon
        self._prev_w_pos = None

    def init(self, obs):
        self._prev_w_pos = None

    def __call__(self, obs, goal):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self._prev_w_pos is None:
            self._prev_w_pos = w_pos
            return 1
        valid_actions = actions_to_goal(self._prev_w_pos, goal)
        if w_action in valid_actions:
            prob =  (1-self._epsilon)/len(valid_actions) + self._epsilon/5
        else:
            prob = self._epsilon/5
        self._prev_w_pos = w_pos
        return prob



class IntermediateGoalModel(Model):
    def __init__(self, intermediate_point):
        self._intermediate_point = intermediate_point
        self._visited = False
        self._prev_w_pos = None

    def init(self, obs):
        self._visited = False
        self._prev_w_pos = None

    def __call__(self, obs, goal):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self._prev_w_pos is None:
            self._prev_w_pos = w_pos
            return 1
        if np.array_equal(w_pos, self._intermediate_point):
            self._visited = True
        if self._visited:
            valid_actions = actions_to_goal(self._prev_w_pos, goal)
        else:
            valid_actions = actions_to_goal(self._prev_w_pos, self._intermediate_point)
        if w_action in valid_actions:
            prob = 1
        else:
            prob = 0
        self._prev_w_pos = w_pos
        return prob
