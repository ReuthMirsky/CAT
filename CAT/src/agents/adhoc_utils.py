from src.agents.agent import AbstractAgent, AgentType
from src import environment as env
import numpy as np
from collections import namedtuple
from enum import Enum
import copy
from src import utils
import warnings


class inference_engine():
    """
    This class helps us do inference on the agent. Given an observation/series of observations we should be able to derive a posterior probability on the type. However, in the current setting, since we only identify the next immediate station as a part of the type and not the entire sequence, we derive posterior probabilities for just that.

    So, basically, we have an obs. This contains the leader agent's position as well as the action it took. From this, we want to be able to infer which station the leader would be going to next.

    We want to do this by feeding the observation to a bunch of dummy agents with different station targets and see which station target explains the action the most. This way we can derive the posterior probabilities.

    We can obtain the set of dummy agents with different station targets by copying the original agent, and setting it's type accordingly.

    We delete and create a new inference engine everytime we finish working on a station
    """
    def __init__(self,tracking_agent,tracking_stations):
        """
        tracking_agent: The agent that we are tracking.
        tracking_stations: The stations that we are considering between. Should be a list/numpy
        """
        self.tracking_agent = tracking_agent.__deepcopy__()
        self.tracking_stations = copy.deepcopy(tracking_stations)
        self.prior = np.ones(len(self.tracking_stations))
        self.prior/=np.sum(self.prior)
        self.current_target = None

    def get_modified_obs(self,previous_obs,current_obs):
        """
        When we calculate likelihood, we are looking to infer from the other agent's decision making process, which involved it being passed an obs and then performing an action. Hence, we need an obs from the previous time-step, which it used to eexecute an action at the end of the time-step (this action is a part of current_obs.allActions)

        Creating a modified_obs simply involves pairing theese two together

        """
        pobs = previous_obs
        cobs = current_obs
        newobs = copy.deepcopy(pobs) #copy previous observations.
        newobs.allActions = cobs.allActions #swap actions.
        return newobs


    def get_likelihood(self,mobs):
        """
        mobs: modified_obs
        possible_station_ids: The possible station_ids whose likelihood we want to measure.

        We want to get the likelihood of seeing the action performed by the leader-agent (contained in obs), give it's position (contained in obs), for each of the possible stations it was targeting

        For this to work, we require a modified_obs, which will include positions from a previous time-step instead of current-time steps. Because, the action was decided when the agent's were in the previous-time step's position and not their current position.

        """
        #First set the tracking_agent's position to the observation reported position.
        obs = mobs
        # self.tracking_agent.pos = obs.allPos[env.LEADER_IDX]

        likelihood_vector = np.zeros(len(self.tracking_stations))
        action_performed = obs.allActions[env.LEADER_IDX]

        # def create_dummy_type(station_id):
        #     """Create a dummy type with only one station to work on"""
        #     dummy_tp = AgentType(1)
        #     dummy_tp.station_order[0] = station_id
        #     return dummy_tp

        # for idx,sttn in enumerate(self.tracking_stations):
        #     dummy_tp = create_dummy_type(sttn)
        #     self.tracking_agent.set_tp(dummy_tp)
        #     action_probs, action = self.tracking_agent.respond(obs)
        #     likelihood_vector[idx] = action_probs[action_performed]

        move = env.ACTIONS_TO_MOVES[action_performed]

        for idx, sttn in enumerate(self.tracking_stations):
            sttn_pos = obs.allPos[obs.stationInd[0] + sttn]
            agent_pos = obs.allPos[env.LEADER_IDX]

            prev_dist = agent_pos.manhattan_dist(sttn_pos)
            curr_dist = (agent_pos + move).manhattan_dist(sttn_pos)

            # When agent moves closer to a station
            if curr_dist < prev_dist:
                likelihood_vector[idx] = 1
            # When agent doesn't move and is on top of a station
            elif curr_dist == prev_dist and agent_pos == sttn_pos:
                likelihood_vector[idx] = 1

        return likelihood_vector

    def inference_step(self,pobs,cobs):
        """
        Should take previous and current obs and return posterior estimate of the next station.
        returns: station to work on.
        """
        mobs = self.get_modified_obs(pobs,cobs)
        ll = self.get_likelihood(mobs)
        (rand_map_idx,ps) = utils.get_MAP(self.prior,ll)
        assert(np.all(ps.shape==ll.shape))
        self.prior = ps

        maxes = np.where(ps == np.amax(ps))[0]

        # This stops changing target when already chosen target is still possible (stops thrashing)
        target_idx = self.current_target
        if self.current_target not in maxes:
            target_idx = rand_map_idx
            self.current_target = rand_map_idx

        certainty = False
        if len(maxes) == 1:
            certainty = True

        return self.tracking_stations[target_idx], certainty



class Knowledge(AgentType):
    """
    Class to represent the knowledge an Adhoc agent has about the other agent. Knowledge here is about the type, specifically, which is why it inehrits the AgentType class.

    The following elements make up knowledge.

    a) Origin - Inference/QA.- Inference is volatile, QA is trustworthy since the agent doesn't lie.
    b) station_order - What is your current guess/estimate of the staion_order in which the adhoc agent is proceeding. If you don't have a estimate, keep it None. It's None.
    c) station_work_status - Does anything in the history indicate whether work has been done at this station or not. If the work has been marked done, then do not bother about editing the station_order's

    """
    origin = Enum('KnowledgeSource',[('Inference',1),('Answer',2)])

    def __init__(self):
        super().__init__(env.N_STATIONS) #Initiaize the underlying type.
        self.station_order = [None for sttn in self.station_order]
        self.knowledge = self.station_order #We call station_order as knowledge
        self.source = [None for sttn in self.knowledge] #We don't have knowledge yet, so it's no source.

    def get_current_job_station(self):
        """
        With the current knowledge, get the current station where work is proceeding/needs to proceed. With this, the adhoc agent can either retrieve the tool if it doesn't have it, or simply go assist the leader agent.

        Note: This overrides the get_current_job_station in the agent.AgentType method.

        In future, this should support returning tool assignments too.

        returns: station_id to work on.
        """

        return super().get_current_job_station()

    def get_current_job_station_idx(self):
        curr_sttn_idx = 0
        while(self.station_work_status[curr_sttn_idx] is AgentType.status.done):
            curr_sttn_idx += 1
        return curr_sttn_idx

    def update_knowledge_from_qa(self,station_order):
        """
        Recieve knowledge from the QA system.

        The QA system is expected to return a station_order that is understood/derived from the answer given. This knowledge is absolute and trusted and true, since the agent doesn't lie.

        We append this knowledge to what we already have, since questions refer to future and not the past, i.e., a question is asked for a dilemma that is in present or in future.
        """

        #seek the current confusing station, which should be the station we are working on right now.

        curr_sttn_idx = self.get_current_job_station_idx()

        #Now append the knowledge.
        for idx in range(0,len(station_order)):
            if idx + curr_sttn_idx < self.n_stations:
                self.station_order[idx+curr_sttn_idx] = station_order[idx]
                self.source[idx+curr_sttn_idx] = Knowledge.origin.Answer

    def update_knowledge_from_inference(self,station):
        """
        Since inference only gives out one station per iteration, we have only one update to do.

        """
#seek the current confusing station, which should be the station we are working on right now.

        curr_sttn_idx = self.get_current_job_station_idx()
        self.station_order[curr_sttn_idx] = station
        self.source[curr_sttn_idx] = Knowledge.origin.Inference
