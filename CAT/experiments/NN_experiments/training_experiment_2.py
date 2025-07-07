import context
from src.acd_utils import ACD_iter2, random_optimal_plan, move_transition, move_actions
from src.environment import ToolFetchingEnvironment
from src.NN_edp_utils import create_model, trainSupervised, trainMC, trainTD
import numpy as np
import sys
import tensorflow as tf
import json
import argparse
tf.enable_eager_execution()



def random_optimal_policy(obs, g, action):
    return random_optimal_plan(obs[0], obs[1], g, action)

def action_probs(obs, g):
    return [random_optimal_policy(obs, g, a) for a in move_actions]
def move_transition2(obs, action, width=20, height=20):
    i,j = move_transition(obs[0], obs[1], action, width, height)
    obs[0] = i
    obs[1] = j
    return obs

def gatherSupervisedData(num_goals, grid_width=20, grid_height=20):
    data = {}
    for _ in range(num_goals):
        g1 = np.random.randint([grid_width, grid_height])
        g2 = np.random.randint([grid_width, grid_height])
        while np.array_equal(g1, g2):
            g2 = np.random.randint([grid_width, grid_height])
        data[tuple(g2), tuple(g1)] = ACD_iter2(g1, g2, random_optimal_plan, move_transition, move_actions, width=grid_width, height=grid_height)
        print(g1, g2)
    return data


def gatherRolloutData(num_goals, grid_width=20, grid_height=20):
    data = {}
    for _ in range(num_goals):
        g = np.random.randint([grid_width, grid_height])
        pos = np.random.randint([grid_width, grid_height])
        traj = []
        while not np.array_equal(pos, g):
            action = np.random.choice(move_actions, p=action_probs(pos, g))
            op = np.array(pos)
            pos = move_transition2(pos, action, grid_width, grid_height)
            traj.append((op, action, np.array(pos)))
        traj.append((np.array(pos), ToolFetchingEnvironment.WORKER_ACTIONS.WORK, None))
        data[tuple(g)] = traj
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='output file')
    parser.add_argument('--num_rollouts', type=int, help='number of rollouts to be used as training data')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs to perform')
    parser.add_argument('--updates_per_epoch', type=int, help='number of times to update for each epoch')
    args = parser.parse_args()
    testSet = gatherSupervisedData(20)
    rollout_data = gatherRolloutData(args.num_rollouts)
    MCTDModel = create_model(6)
    for k in range(args.num_epochs//2):
        print(k)
        trainMC(MCTDModel, rollout_data, 1, random_optimal_policy, updatesPerEpoch=args.updates_per_epoch)
        results = {}
        results['MCTD']= []
        out = {}
        for g1, g2 in testSet:
            s = testSet[g1,g2]
            for i in range(len(s)):
                for j in range(len(s[i])):
                    state = np.array([[i, j] + list(g1) + list(g2)])
                    results['MCTD'].append((s[i][j] - MCTDModel(state).numpy()[0][0])**2)
        out['graph 1'] = results
        with open(args.output + '/results' + str(k) + '.json', 'w') as f:
            json.dump(out, f)
    for l in range(int(np.ceil(args.num_epochs/2))):
        print(k+l)
        trainTD(MCTDModel, rollout_data, 1, random_optimal_policy, updatesPerEpoch=args.updates_per_epoch)
        results = {}
        results['MCTD']= []
        out = {}
        for g1, g2 in testSet:
            s = testSet[g1,g2]
            for i in range(len(s)):
                for j in range(len(s[i])):
                    state = np.array([[i, j] + list(g1) + list(g2)])
                    results['MCTD'].append((s[i][j] - MCTDModel(state).numpy()[0][0])**2)
        out['graph 1'] = results
        with open(args.output + '/results' + str(k+l) + '.json', 'w') as f:
            json.dump(out, f)
