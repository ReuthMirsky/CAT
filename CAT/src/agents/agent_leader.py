import numpy as np
import matplotlib.pyplot as plt
from src.agents import agent
from src.astar import pyastar
from collections import namedtuple
from src import environment as env
from src import utils
import copy

#TODO:
"""
Leader:

Modifications from original:

Init:

1) Generate a random order of stations and tools.

At each step:

1) If a question is asked, pause and answer it.

2) Else:

If neighboring a station:

    If station task not completed:

        If next agent is here:

            Mark the task finished

        else

            Wait until next agent arrives.

 Else

        Navigate to the next station.

3) Repeat until all stations are marked done.

"""


# leader_agent_state = namedtuple('LeaderAgentState','type pos target')


class agent_leader(agent.AbstractAgent):
    def __init__(self,pos,tp=None,path=[]):
        super().__init__(pos,tp)
        self.pos = pos
        self.name = self.name+'_leader_'+str(self.id)
        self.path = path.copy()

        # If type is provided, set agent to that type
        if tp is not None:
            assert isinstance(tp, agent.AgentType)
            self.tp = tp.__copy__()
            self.__target = self.tp.get_current_job_station()
        # If type is not provided, set agent to random type
        else:
            self.tp = agent.AgentType(env.N_STATIONS) #Initializing type from settings derived from global_defs.
            target = self.tp.get_current_job_station()
            self.__target = target

    def respond(self,observation: env.OBS)-> tuple:
        obs = observation
        target = self.tp.get_current_job_station()
        self.__target = target
        target_pos = obs.allPos[obs.stationInd[target]]
        # obstacles = copy.deepcopy(obs.allPos)
        # obstacles.remove(self.pos)
        obstacles = []

        if len(self.path):
           desired_action = self.path.pop(0)
        else:
            desired_action = None
            if utils.is_neighbor(self.pos,target_pos):
                #We are neighboring the station we want to work at.
                if utils.is_neighbor(target_pos,obs.allPos[env.ADHOC_IDX]):
                    #Meaning, if the other agent is also neighoring the station, execute the work action
                    # Environment _proposal_check() checks if adhoc has the right tool
                    desired_action = env.Actions.WORK
                    #now, if the other agent also executed a WORK action, and happens to have the same tool, then we can safely move to the next station. This change will happen if the environment approves the work action, since, for the action to run, the agent needs a way to know if the other agent also has thetool to operate in this station. This tool checking will be done by the environment.
                else:
                    #Else, just wait.
                    desired_action = env.Actions.NOOP
            else:
                #Meaning we yet have to reach our target station.
                desired_action = None

        proposal = utils.generate_proposal(self.pos,target_pos,obstacles,desired_action)
        return proposal

    def act(self,proposal,decision):
        if decision is True:
            _,action = proposal
            #Then we are allowed to execute the action.
            #First, apply the movement.
            self.pos += env.ACTIONS_TO_MOVES[action]
            if action == env.Actions.WORK:
                #Which means we were approved to go ahead and do the work action, because the other agent had the right tool with it. It's time to move onto the next station.
                self.tp.set_status(self.tp.get_current_job_station_idx()) #mark the station's status as done.
        else:
            pass

    def set_tp(self,tp):
        self.tp = tp

    def __repr__(self):
        st = '{}_at_{}_withtp_{}_goingto_{}'.format(self.name,self.pos,self.tp,self.__target)
        return st

    def __copy__(self):
        """
        Make a copy of the agent. Copying the following is enough to make a good copy of the agent. We don't differentiate between deepcopy and copy. All are the same.
        """
        new_tp = self.tp.__deepcopy__()
        new_pos = copy.deepcopy(self.pos)
        new_agent = agent_leader(new_pos,new_tp)
        return new_agent

    def __deepcopy__(self):
        return self.__copy__()

    def make_identical_to(self,new_agent):
        """
        This makes the agents' identical to each other.
        """

        self.tp = copy.deepcopy(new_agent.tp)
        self.pos = copy.deepcopy(new_agent.pos)
        return True

    # def __getstate__(self):
    #
    #     return leader_agent_state
    #
    # def __setstate__(self):
    #     pass
