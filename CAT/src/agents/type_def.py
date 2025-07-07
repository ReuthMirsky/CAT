import numpy as np
import matplotlib.pyplot as plt
from src import global_defs as global_defs




#Type definitions go here. A type is defined by the mapping from state/observation to distribution over action probabilities.

#For our current purposes, agent positions should be enough.

class type_1():
    def __init__(self,n_stations=global_defs.N_STATIONS)
        self.n_stations = global_defs.N_STATIONS
        
    def get_actionProbs(self,obs,desired_station_order,curr_station_id):
        """
        obs: Observation as defined in global_defs
        desired_station_order: A list of station indices describing the order of traversal of these stations.
        curr_station_id: Index of the most recent station covered in the above list.

        Eg: If we chose to cover stations in the order 3,1,2 and we have most recently finished station 1, then desired_station_order = [3,1,2] and curr_station_id = 2

        For our current case, it should be really simple since all we care is going from point A to point B, then once we arrive at a station, we simply execute the action to begin work.
        """

        
        
        
        
