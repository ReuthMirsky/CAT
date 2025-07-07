import numpy as np


# Returns list of valid actions that brings fetcher closer to all tools
def get_valid_actions(obs, agent):
    w_pos, f_pos, s_pos, t_pos, f_tool, w_action, f_action, answer = obs

    valid_actions = np.array([True] * 4) # NOOP is always valid
    for stn in range(len(s_pos)):
        if agent.probs[stn] == 0:
            continue
        tool_valid_actions = np.array([True] * 4)
        if f_pos[0] <= t_pos[stn][0]:
            tool_valid_actions[1] = False # Left
        if f_pos[0] >= t_pos[stn][0]:
            tool_valid_actions[0] = False # Right
        if f_pos[1] >= t_pos[stn][1]:
            tool_valid_actions[2] = False # Down
        if f_pos[1] <= t_pos[stn][1]:
            tool_valid_actions[3] = False # Up

        valid_actions = np.logical_and(valid_actions, tool_valid_actions)

    return valid_actions
