"""
This file contains code relevant to the Tool Fetching Domain Environment
"""
import gym
from enum import IntEnum
import numpy as np
import copy

def uniform_cost(state, query):
    return 1


class ToolFetchingEnvironment(gym.Env):
    """
    Tool Fetching Domain Environment
    This is a multiagent environment consisting of a worker traveling to some station and a fetcher attempting to bring a tool to the worker

    State Space
    ---------------------------------------------------------------------------------------------------------
    Worker: Tuple(Worker Position, Fetcher Position, Station Positions, Tool Positions, 
        Fetcher's Tool, Worker's Last Action, Fetcher's Last Action, Worker's Goal)
    Fetcher: Tuple(Worker Position, Fetcher Position, Station Positions, Tool Positions, 
        Fetcher's Tool, Worker's Last Action, Fetcher's Last Action, Worker's answer to Query(if any))
    ----------------------------------------------------------------------------------------------------------

    Action Space
    ----------------------------------------------------------------------------------------------------------
    Worker: Left, Right, Up Down Noop Work
    Fetcher: Left Right Up Down Noop Query Pickup
    When Querying: Fetcher passes in list of stations (is your station in this list)
    When Picking Up, Fetcher Passes in target tool
    ----------------------------------------------------------------------------------------------------------

    Reward
    ----------------------------------------------------------------------------------------------------------
    Both agents receive -1 every timestep the simulation continues and 0 on termination
    ----------------------------------------------------------------------------------------------------------
    """
    FETCHER_ACTIONS = IntEnum('FETCHER_Actions', 'RIGHT LEFT UP DOWN NOOP QUERY PICKUP', start=0)
    FETCHER_ACTION_VALUES = set(a.value for a in FETCHER_ACTIONS)
    WORKER_ACTIONS = IntEnum('WORKER_Actions', 'RIGHT LEFT UP DOWN NOOP WORK', start=0)
    WORKER_ACTION_VALUES = set(a.value for a in WORKER_ACTIONS)
    def __init__(self, fetcher_pos, worker_pos, stn_pos, tool_pos, worker_goal, width=10, height=10, cost_fun=uniform_cost):
        assert len(stn_pos) == len(tool_pos)
        assert worker_goal >= 0 and worker_goal < len(stn_pos)
        self.width = width
        self.height = height
        self.f_pos = copy.deepcopy(fetcher_pos)
        self.w_pos = copy.deepcopy(worker_pos)
        self.s_pos = copy.deepcopy(stn_pos)
        self.t_pos = copy.deepcopy(tool_pos)
        self.curr_f_pos = copy.deepcopy(fetcher_pos)
        self.curr_w_pos = copy.deepcopy(worker_pos)
        self.curr_t_pos = copy.deepcopy(tool_pos)
        self.f_tool = None
        self.w_goal = worker_goal

        self.viewer = None
        
        self._cost_fun = cost_fun

    def make_fetcher_obs(self, w_action, f_action, answer=None):
        return copy.deepcopy((self.curr_w_pos, self.curr_f_pos, self.s_pos, self.curr_t_pos, self.f_tool, w_action, f_action, answer))

    def make_worker_obs(self, w_action, f_action):
        return copy.deepcopy((self.curr_w_pos, self.curr_f_pos, self.s_pos, self.curr_t_pos, self.f_tool, w_action, f_action, self.w_goal))

    def _apply_worker_action(self, worker_action):
        if worker_action == ToolFetchingEnvironment.WORKER_ACTIONS.RIGHT:
            self.curr_w_pos += np.array([1,0])
            self.curr_w_pos[0] = min(self.curr_w_pos[0], self.width-1)
        elif worker_action == ToolFetchingEnvironment.WORKER_ACTIONS.LEFT:
            self.curr_w_pos -= np.array([1,0])
            self.curr_w_pos[0] = max(self.curr_w_pos[0], 0)
        elif worker_action == ToolFetchingEnvironment.WORKER_ACTIONS.UP:
            self.curr_w_pos += np.array([0,1])
            self.curr_w_pos[1] = max(self.curr_w_pos[1], 0)
        elif worker_action == ToolFetchingEnvironment.WORKER_ACTIONS.DOWN:
            self.curr_w_pos -= np.array([0,1])
            self.curr_w_pos[1] = min(self.curr_w_pos[1], self.height-1)

    def _apply_fetcher_action(self, f_action):
        fetcher_action, fetcher_details = f_action
        if fetcher_action == ToolFetchingEnvironment.FETCHER_ACTIONS.RIGHT:
            self.curr_f_pos += np.array([1,0])
            self.curr_f_pos[0] = min(self.curr_f_pos[0], self.width-1)
        elif fetcher_action == ToolFetchingEnvironment.FETCHER_ACTIONS.LEFT:
            self.curr_f_pos -= np.array([1,0])
            self.curr_f_pos[0] = max(self.curr_f_pos[0], 0)
        elif fetcher_action == ToolFetchingEnvironment.FETCHER_ACTIONS.UP:
            self.curr_f_pos += np.array([0,1])
            self.curr_f_pos[1] = max(self.curr_f_pos[1], 0)
        elif fetcher_action == ToolFetchingEnvironment.FETCHER_ACTIONS.DOWN:
            self.curr_f_pos -= np.array([0,1])
            self.curr_f_pos[1] = min(self.curr_f_pos[1], self.height-1)
        elif fetcher_action == ToolFetchingEnvironment.FETCHER_ACTIONS.PICKUP:
            assert fetcher_details >= 0 and fetcher_details < len(self.curr_t_pos)
            if np.array_equal(self.curr_t_pos[fetcher_details], self.curr_f_pos):
                self.f_tool = fetcher_details


    def step(self, action_n):
        worker_action = action_n[0]
        fetcher_action, fetcher_details = action_n[1]
        assert worker_action in ToolFetchingEnvironment.WORKER_ACTION_VALUES
        assert fetcher_action in ToolFetchingEnvironment.FETCHER_ACTION_VALUES
        if fetcher_action == ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY:
            answer = self.answer_query(fetcher_details)
            obs_n = np.array([self.make_worker_obs(ToolFetchingEnvironment.WORKER_ACTIONS.NOOP, ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY), self.make_fetcher_obs(ToolFetchingEnvironment.WORKER_ACTIONS.NOOP, ToolFetchingEnvironment.FETCHER_ACTIONS.QUERY, answer)])
            reward = -self._cost_fun(obs_n, fetcher_details)
            reward_n = np.array([reward,reward])
            done_n = np.array([False, False])
            info_n = np.array([{}, {}])
            return obs_n, reward_n, done_n, info_n


        self._apply_worker_action(worker_action)
        self._apply_fetcher_action(action_n[1])



        if self.f_tool is not None:
            self.curr_t_pos[self.f_tool] = self.curr_f_pos



        if worker_action == ToolFetchingEnvironment.WORKER_ACTIONS.WORK:
            goal_pos = self.s_pos[self.w_goal]
            tool_pos = self.curr_t_pos[self.w_goal]
            if np.array_equal(self.curr_w_pos,  goal_pos) and np.array_equal(self.curr_w_pos, tool_pos):
                obs_n = np.array([self.make_worker_obs(ToolFetchingEnvironment.WORKER_ACTIONS.WORK, fetcher_action), self.make_fetcher_obs(ToolFetchingEnvironment.WORKER_ACTIONS.WORK, fetcher_action)])
                reward_n = np.array([0,0])
                done_n = np.array([True, True])
                info_n = np.array([{}, {}])
                return obs_n, reward_n, done_n, info_n



        obs_n = np.array([self.make_worker_obs(worker_action, fetcher_action), self.make_fetcher_obs(worker_action, fetcher_action)])
        reward_n = np.array([-1,-1])
        done_n = np.array([False, False])
        info_n = np.array([{}, {}])

        return obs_n, reward_n, done_n, info_n


    def answer_query(self, query):
        if self.w_goal in query:
            return True
        return False

    def reset(self):
        self.curr_f_pos = np.array(self.f_pos)
        self.curr_w_pos = np.array(self.w_pos)
        self.curr_t_pos = np.array(self.t_pos)
        self.f_tool = None
        obs_n = np.array([self.make_worker_obs(ToolFetchingEnvironment.WORKER_ACTIONS.NOOP, ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP), self.make_fetcher_obs(ToolFetchingEnvironment.WORKER_ACTIONS.NOOP, ToolFetchingEnvironment.FETCHER_ACTIONS.NOOP)])
        return obs_n

    def render(self):

        colors = {
                'green':[0,1,0],
                'blue':[0,0,1],
                'violet':[1,0,1],
                'yellow':[1,1,0]
                }

        color_keys = list(colors.keys())

        screen_width = 600
        screen_height = 600

        horz_line_spacing = screen_width//self.width
        vert_line_spacing = screen_height//self.height

        worker_x,worker_y = self.curr_w_pos
        fetcher_x,fetcher_y = self.curr_f_pos

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            for i in range(self.height):
                line = rendering.Line(start=(0,vert_line_spacing*i), end=(screen_width, vert_line_spacing*i))
                self.viewer.add_geom(line)
            for i in range(self.width):
                line = rendering.Line(start=(horz_line_spacing*i, 0), end=(horz_line_spacing*i, screen_height))
                self.viewer.add_geom(line)
            self.tool_transforms = []
            for i in range(len(self.curr_t_pos)):
                color = color_keys[i%len(color_keys)]
                station_x, station_y = self.s_pos[i]

                station_shape = rendering.make_polygon([(-horz_line_spacing/2,-vert_line_spacing/2), 
                    (-horz_line_spacing/2,vert_line_spacing/2), 
                    (horz_line_spacing/2,vert_line_spacing/2), 
                    (horz_line_spacing/2,-vert_line_spacing/2)])
                station_transform = rendering.Transform(translation = (station_x*horz_line_spacing+horz_line_spacing/2, station_y*vert_line_spacing + vert_line_spacing/2) )
                station_shape.add_attr(station_transform)
                station_shape.set_color(*colors[color])
                self.viewer.add_geom(station_shape)


                tool_shape = rendering.make_polygon([(-horz_line_spacing/2,-vert_line_spacing/2), 
                    (-horz_line_spacing/2,vert_line_spacing/2), 
                    (horz_line_spacing/2,vert_line_spacing/2), 
                    (horz_line_spacing/2,-vert_line_spacing/2)])
                tool_transform = rendering.Transform(scale=(0.25,1))
                tool_shape.add_attr(tool_transform)
                tool_shape.set_color(*colors[color])
                self.viewer.add_geom(tool_shape)
                self.tool_transforms.append(tool_transform)


            worker_shape = rendering.make_circle()
            self.worker_transform = rendering.Transform(scale=(horz_line_spacing/20, vert_line_spacing/20))
            worker_shape.add_attr(self.worker_transform)
            worker_shape.set_color(0,0,0)
            self.viewer.add_geom(worker_shape)
            fetcher_shape = rendering.make_circle()
            self.fetcher_transform = rendering.Transform(scale=(horz_line_spacing/20, vert_line_spacing/20))
            fetcher_shape.add_attr(self.fetcher_transform)
            fetcher_shape.set_color(1,0,0)
            self.viewer.add_geom(fetcher_shape)

                

        self.worker_transform.set_translation(worker_x*horz_line_spacing + horz_line_spacing/2, worker_y*vert_line_spacing + vert_line_spacing/2)
        self.fetcher_transform.set_translation(fetcher_x*horz_line_spacing + horz_line_spacing/2, fetcher_y*vert_line_spacing + vert_line_spacing/2)
        for i,transform in enumerate(self.tool_transforms):
            tool_x,tool_y = self.curr_t_pos[i]
            transform.set_translation(tool_x*horz_line_spacing + horz_line_spacing/2, tool_y*vert_line_spacing + vert_line_spacing/2)
        


        return self.viewer.render(False)


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class ToolFetchingEnvironmentWithObstacles(ToolFetchingEnvironment):
    def __init__(self, fetcher_pos, worker_pos, stn_pos, tool_pos, worker_goal, obst_pos, width=10, height=10):
        super().__init__(fetcher_pos, worker_pos, stn_pos, tool_pos, worker_goal, width, height)
        self._obst_pos = obst_pos


    def _apply_worker_action(self, w_action):
        old_w_pos = np.array(self.curr_w_pos)
        super()._apply_worker_action(w_action)
        if self.curr_w_pos in self._obst_pos:
            self.curr_w_pos = old_w_pos


    def _apply_fetcher_action(self, f_action):
        old_f_pos = np.array(self.curr_f_pos)
        super()._apply_fetcher_action(f_action)
        if self.curr_f_pos in self._obst_pos:
            self.curr_f_pos = old_f_pos


    def render(self):
        super().render()
        from gym.envs.classic_control import rendering
        screen_width = 600
        screen_height = 600

        horz_line_spacing = screen_width//self.width
        vert_line_spacing = screen_height//self.height
        for o in self._obst_pos:
            obst_x, obst_y = o
            obst_shape = rendering.make_polygon([(-horz_line_spacing/2,-vert_line_spacing/2), 
                (-horz_line_spacing/2,vert_line_spacing/2), 
                (horz_line_spacing/2,vert_line_spacing/2), 
                (horz_line_spacing/2,-vert_line_spacing/2)])
            obst_transform = rendering.Transform(translation = (obst_x*horz_line_spacing+horz_line_spacing/2, obst_y*vert_line_spacing + vert_line_spacing/2) )
            obst_shape.add_attr(obst_transform)
            self.viewer.add_geom(obst_shape)
        return self.viewer.render(False)




