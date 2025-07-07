"""
This file contains relevant base code for agent Policies and Specific code for Worker's policy
"""
import numpy as np
from src.environment import ToolFetchingEnvironment

class Policy:
    """
    Base Polciy class, __call__ should map observation to action
    """
    def __call__(self, obs):
        raise NotImplementedError

    def reset(self):
        pass

class RandomWorkerPolicy(Policy):
    """
    Random Worker Policy, picks a optimal action randomly (effectively follows a random optimal route to goal)
    """
    def __init__(self):
        self.last_w_pos = None
    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, goal = obs
        goal_pos = s_pos[goal]
        if np.array_equal(goal_pos, w_pos):
            return ToolFetchingEnvironment.WORKER_ACTIONS.WORK
        actions = []
        if w_pos[0] < goal_pos[0]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT)
        elif w_pos[0] > goal_pos[0]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.LEFT)

        if w_pos[1] > goal_pos[1]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.DOWN)
        elif w_pos[1] < goal_pos[1]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.UP)

        return np.random.choice(actions)

class PlanPolicy(Policy):
    """
    Plan Policy, taks a plan as initialization and follows it through simulation, useful for testing
    """
    def __init__(self, plan):
        self._plan = plan
        self._step = 0

    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, goal = obs
        if f_action != ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY:
            action = self._plan[min(self._step, len(self._plan)-1)]
            self._step += 1
            return action
        return self._plan[min(self._step-1, len(self._plan)-1)]

    def reset(self):
        self._step = 0

    @property
    def step(self):
        return self._step

    @property
    def plan(self):
        return self._plan

class SubOptimalWorker(RandomWorkerPolicy):
    def __init__(self, epsilon):
        super().__init__()
        self._epsilon = epsilon
        self._rand_actions = [
                ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT,
                ToolFetchingEnvironment.WORKER_ACTIONS.LEFT,
                ToolFetchingEnvironment.WORKER_ACTIONS.DOWN,
                ToolFetchingEnvironment.WORKER_ACTIONS.UP,
                ToolFetchingEnvironment.WORKER_ACTIONS.WORK]

    def __call__(self, obs):
        normal_action = super().__call__(obs)
        if np.random.random() < self._epsilon:
            return np.random.choice(self._rand_actions)
        return normal_action

class IntermediatePointPolicy(RandomWorkerPolicy):
    def __init__(self, intermediate_point):
        super().__init__()
        self._intermediate_point = intermediate_point
        self._visited = False

    def reset(self):
        self._visited = False

    def __call__(self, obs):
        normal_action = super().__call__(obs)
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, goal = obs
        if np.array_equal(w_pos, self._intermediate_point):
            self._visited = True
        if self._visited:
            return normal_action
        actions = []
        if w_pos[0] < self._intermediate_point[0]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT)
        elif w_pos[0] > self._intermediate_point[0]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.LEFT)

        if w_pos[1] > self._intermediate_point[1]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.DOWN)
        elif w_pos[1] < self._intermediate_point[1]:
            actions.append(ToolFetchingEnvironment.WORKER_ACTIONS.UP)

        return np.random.choice(actions)
