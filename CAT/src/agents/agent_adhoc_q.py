"""
This file contains relevant code for Fetcher's Policies
"""
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
from src.agents.query_policies import never_query
from src.agents.agent_utils import get_valid_actions





class FetcherQueryPolicy(Policy):
    """
    Basic Fetcher Policy for querying, follows query_policy function argument (defaults to never query)
    Assumes all tools are in same location
    """
    def __init__(self, query_policy=never_query, prior=None, agent_model = None):
        self.query_policy = query_policy
        self._prior = prior
        self.probs = copy.deepcopy(self._prior)
        self.query = None
        self.prev_w_pos = None
        self.wcd = None
        self._agent_model = agent_model
        self._log_probs = None


    def reset(self):
        self.probs = copy.deepcopy(self._prior)
        self.query = None
        self.prev_w_pos = None
        self.wcd = None

    def _init_wcd(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        self.wcd = np.zeros((len(s_pos), len(s_pos)), dtype=np.int64)
        for i,g1 in enumerate(s_pos):
            for j,g2 in enumerate(s_pos):
                print(i,j)
                if i==j: 
                    continue
                if i > j:
                    self.wcd[i][j] = self.wcd[j][i]
                    continue
                self.wcd[i][j] = fast_wcd(w_pos, [g1, g2])
        print(self.wcd)


    def make_inference(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self._agent_model is not None:
            if self.prev_w_pos is None:
                return
            #TODO implement
            if self._log_probs is None:
                self._log_probs = np.log(self.probs)
            for i, stn in enumerate(s_pos):
                #self.probs *= self._agent_model(self.prev_w_pos, stn, w_action)
                self._log_probs[i] += np.log(self._agent_model(self.prev_w_pos, stn, w_action))
            #print("before:", self.probs)
            #print(self._log_probs)
            self.probs = np.exp(self._log_probs)
            self.probs /= np.sum(self.probs)
            #print("after:", self.probs)
            #input()
        else:
            if self.prev_w_pos is None:
                return
            if w_action == ToolFetchingEnvironment.WORKER_ACTIONS.WORK:
                for i,stn in enumerate(s_pos):
                    if not np.array_equal(stn, self.prev_w_pos):
                        self.probs[i] = 0
            elif w_action == ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT:
                for i,stn in enumerate(s_pos):
                    if stn[0] <= self.prev_w_pos[0]:
                        self.probs[i] = 0
            elif w_action == ToolFetchingEnvironment.WORKER_ACTIONS.LEFT:
                for i,stn in enumerate(s_pos):
                    if stn[0] >= self.prev_w_pos[0]:
                        self.probs[i] = 0
            elif w_action == ToolFetchingEnvironment.WORKER_ACTIONS.DOWN:
                for i,stn in enumerate(s_pos):
                    if stn[1] >= self.prev_w_pos[1]:
                        self.probs[i] = 0
            elif w_action == ToolFetchingEnvironment.WORKER_ACTIONS.UP:
                for i,stn in enumerate(s_pos):
                    if stn[1] <= self.prev_w_pos[1]:
                        self.probs[i] = 0

            self.probs /= np.sum(self.probs)


    def action_to_goal(self, pos, goal):
        actions = []
        if pos[0] < goal[0]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT)
        elif pos[0] > goal[0]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT)
        if pos[1] > goal[1]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN)
        elif pos[1] < goal[1]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.UP)
        if len(actions) == 0:
            return ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP
        return np.random.choice(actions)


    def __call__(self, obs):
        #if self.wcd is None:
        #    self._init_wcd(obs)
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.probs is None:
            self.probs = np.ones(len(s_pos))
            self.probs /= np.sum(self.probs)
        if self._log_probs is None:
            self._log_probs = np.log(self.probs)
        if answer is not None:
            if answer:
                for stn in range(len(s_pos)):
                    if stn not in self.query:
                        self.probs[stn] = 0
                        self._log_probs[stn] = float('-inf')
            else:
                for stn in self.query:
                    self.probs[stn] = 0
                    self._log_probs[stn] = float('-inf')
            self.probs /= np.sum(self.probs)
        else:
            self.make_inference(obs)

        self.prev_w_pos = np.array(w_pos)

        self.query = self.query_policy(obs, self)
        if self.query is not None:
            return ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY, self.query

        if np.max(self.probs) < 1:
            #dealing with only one tool position currently
            if np.array_equal(f_pos, t_pos[0]):
                return ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP, None
            else:
                return self.action_to_goal(f_pos, t_pos[0]), None
        else:
            if f_tool != np.argmax(self.probs):
                if np.array_equal(f_pos, t_pos[0]):
                    return ToolFetchingEnvironment.FETCHER_ACTIONS.PICKUP, np.argmax(self.probs)
                else:
                    return self.action_to_goal(f_pos, t_pos[0]), None
            return self.action_to_goal(f_pos, s_pos[np.argmax(self.probs)]), None


class FetcherAltPolicy(FetcherQueryPolicy):
    """
    More Complicated Fetcher Policy, allows for multiple tool locations
    """
    def __call__(self, obs):
        #if self.wcd is None:
        #    self._init_wcd(obs)
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.probs is None:
            self.probs = np.ones(len(s_pos))
            self.probs /= np.sum(self.probs)
        if self._log_probs is None:
            self._log_probs = np.log(self.probs)
        if answer is not None:
            if answer:
                for stn in range(len(s_pos)):
                    if stn not in self.query:
                        self.probs[stn] = 0
                        self._log_probs[stn] = float('-inf')
            else:
                for stn in self.query:
                    self.probs[stn] = 0
                    self._log_probs[stn] = float('-inf')
            self.probs /= np.sum(self.probs)
        else:
            self.make_inference(obs)

        self.prev_w_pos = np.array(w_pos)

        # One station already guaranteed. No querying needed.
        if np.max(self.probs) == 1:
            target = np.argmax(self.probs)
            if f_tool != target:
                if np.array_equal(f_pos, t_pos[target]):
                    return ToolFetchingEnvironment.FETCHER_ACTIONS.PICKUP, target
                else:
                    return self.action_to_goal(f_pos, t_pos[target]), None
            return self.action_to_goal(f_pos, s_pos[target]), None

        self.query = self.query_policy(obs, self)
        if self.query is not None:
            return ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY, self.query

        valid_actions = get_valid_actions(obs, self)

        if np.any(valid_actions):
            #print(valid_actions)
            p = valid_actions / np.sum(valid_actions)
            action_idx = np.random.choice(np.arange(4), p=p)
            return ToolFetchingEnvironment.FETCHER_ACTIONS(action_idx), None
        else:
            return ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP, None
        
class FetcherAltPolicy2(FetcherQueryPolicy):
    """
    More Complicated Fetcher Policy, allows for multiple tool locations
    """
    def __init__(self, query_policy=never_query, prior=None, wcd=None, agent_model = None):
        super().__init__(query_policy, prior, agent_model)
        self.wcd = wcd
        self.time = 0

    def reset(self):
        super().reset()
        self.time = 0

    def __call__(self, obs):
        #if self.wcd is None:
        #    self._init_wcd(obs)
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if self.wcd is None:
            self.wcd = {}
            for g1 in range(len(t_pos)):
                for g2 in range(len(t_pos)):
                    if g1 == g2:
                        continue
                    self.wcd[g1,g2] = fast_wcd(f_pos, [t_pos[g1], t_pos[g2]])
        if self.probs is None:
            self.probs = np.ones(len(s_pos))
            self.probs /= np.sum(self.probs)
        if self._log_probs is None:
            self._log_probs = np.log(self.probs)
        if answer is not None:
            if answer:
                for stn in range(len(s_pos)):
                    if stn not in self.query:
                        self.probs[stn] = 0
                        self._log_probs[stn] = float('-inf')
            else:
                for stn in self.query:
                    self.probs[stn] = 0
                    self._log_probs[stn] = float('-inf')
            self.probs /= np.sum(self.probs)
        else:
            self.make_inference(obs)

        self.prev_w_pos = np.array(w_pos)

        # One station already guaranteed. No querying needed.
        if np.max(self.probs) == 1:
            target = np.argmax(self.probs)
            if f_tool != target:
                if np.array_equal(f_pos, t_pos[target]):
                    self.time += 1
                    return ToolFetchingEnvironment.FETCHER_ACTIONS.PICKUP, target
                else:
                    self.time += 1
                    return self.action_to_goal(f_pos, t_pos[target]), None
            self.time += 1
            return self.action_to_goal(f_pos, s_pos[target]), None

        self.query = self.query_policy(obs, self)
        if self.query is not None:
            return ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY, self.query


        valid_actions = get_valid_actions(obs, self)

        if np.any(valid_actions):
            #print(valid_actions)
            p = valid_actions / np.sum(valid_actions)
            action_idx = np.random.choice(np.arange(4), p=p)
            self.time += 1
            return ToolFetchingEnvironment.FETCHER_ACTIONS(action_idx), None
        else:
            print("Should not happen unless never query")
            print(self.probs)
            return ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP, None

class FetcherAgentTypePolicy(Policy):
    def __init__(self, agent_classifier, query_policy=never_query):
        self._query_policy = query_policy
        self._agent_classifier = agent_classifier
        self._probs = None
        self._full_probs = None

    def reset(self):
        self._agent_classifier.reset()
        self._probs = None
        self._full_probs = None

    def make_inference(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        num_g = len(s_pos)
        num_a = self._agent_classifier.num_agent_types
        if not self._agent_classifier.initialized:
            self._agent_classifier.init(obs)
        if self._full_probs is None:
            self._full_probs = np.ones((num_g, num_a))
            self._full_probs /= np.sum(self._full_probs)
        if self._probs is None:
            self._probs = np.empty(num_g)
        self._full_probs *= self._agent_classifier(obs)
        self._full_probs /= np.sum(self._full_probs)

        if not any(p > 0 for p in self._full_probs.flatten()):
            self._full_probs = np.ones((num_g, num_a))
            self._full_probs /= np.sum(self._full_probs)

        for i in range(num_g):
            self._probs[i] = np.sum(self._full_probs[i, :])



    def _action_to_goal(self, pos, goal):
        actions = []
        if pos[0] < goal[0]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT)
        elif pos[0] > goal[0]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT)
        if pos[1] > goal[1]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN)
        elif pos[1] < goal[1]:
            actions.append(ToolFetchingEnvironment.FETCHER_ACTIONS.UP)
        if len(actions) == 0:
            return ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP
        return np.random.choice(actions)


    def __call__(self, obs):
        w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs
        if answer is not None:
            raise NotImplementedError
        else:
            self.make_inference(obs)
        goal = np.argmax(self._probs)
        if f_tool == goal:
            return self._action_to_goal(f_pos, s_pos[goal]), None
        else:
            if np.array_equal(f_pos, t_pos[goal]):
                return ToolFetchingEnvironment.FETCHER_ACTIONS.PICKUP, goal
            else:
                return self._action_to_goal(f_pos, t_pos[goal]), None
