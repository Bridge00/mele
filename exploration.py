import numpy as np
import os
import argparse
from agents import *
from scenario_utils import *
#from matterport_environment import *
from torch.utils.data import DataLoader, TensorDataset
import pickle
from concurrent.futures import ThreadPoolExecutor
import json
from collections import defaultdict

# def getObservation(subAgent):

#     subAgent.explorationStep(timestep, 1000, args.SCAN_ID)
#     filename = f'{subAgent.agent_id}_id_{subAgent.model}_model_{subAgent.behavior}_exploration.pkl'
#     with open(os.path.join(folder_name, filename, 'wb')) as f:
#         pickle.dump(subAgent.observation_dict, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Multi-LLM behavior')

    parser.add_argument('--num_questions', '-q', type=int, default= 100, metavar='N',
                        help='number of questions')
    
    parser.add_argument('--dataset', '-d', type=str, default= 'embodied', metavar='N',
                        help='dataset used')

    parser.add_argument('--scenario', '-s', type=int, default= 0, metavar='N',
                    help='scenario for subordinate agent') 
    
    parser.add_argument('--SCAN_ID', '-scan', type=str, default= '2azQ1b91cZZ', metavar='N',
                    help='scan_ID') 
    
    # parser.add_argument('--scans', '-scan', nargs='*', type=str, default= ['2azQ1b91cZZ'], metavar='N',
    #                 help='scan_ID') 
    parser.add_argument('--total_timesteps', '-tt', type=int, default= 10, metavar='N',
                    help='total_timesteps') 


    args = parser.parse_args() 

    # central = centralAgent(scenarios[args.scenario]['central_agent_model'])

    subAgents = []

    submodels = exploration_scenarios[args.scenario]['subordinate_agent_models']
    behaviors = exploration_scenarios[args.scenario]['exploration_behaviors']

    
    with open('scan_to_node_to_room.pkl', 'rb') as f:
        SCAN_TO_NODE_TO_ROOM = pickle.load(f)
    # print(G)


    # G = load_nav_graph(args.SCAN_ID)
    # print(args.SCAN_ID, len(G.nodes()), len(set([SCAN_TO_NODE_TO_ROOM[args.SCAN_ID][node] for node in SCAN_TO_NODE_TO_ROOM[args.SCAN_ID]])))

    for scan in ['2azQ1b91cZZ',  'EU6Fwq7SyZv',  'TbHJrupSAjP',  'Z6MFQCViBuw',  'x8F5xyUWy9e', '8194nk5LbLH',  'QUCTc6BB5sX',  'X7HyMhZNoso',  'oLBMNvg9in8', 'zsNo4HB9uLZ']:
        G = load_nav_graph(scan)
        print(scan, len(G.nodes()), len(set([SCAN_TO_NODE_TO_ROOM[args.SCAN_ID][node] for node in SCAN_TO_NODE_TO_ROOM[args.SCAN_ID]])))



    # folder_name = f'observation_dictionaries/glip/{args.SCAN_ID}/'
    # os.makedirs(folder_name, exist_ok= True)
    # print(folder_name)
    # behavior_count = defaultdict(int)
    # for (index, (model, behavior)) in enumerate(zip(submodels, behaviors)):
    #     behavior_folder = os.path.join(folder_name, behavior)
    #     os.makedirs(behavior_folder, exist_ok=True)
    #     subAgents.append(subAgent(len(behavior_folder) + behavior_count[behavior], model, behavior, np.random.choice(G.nodes())))
    #     behavior_count[behavior] += 1
    # # print(subAgents)

    # for timestep in range(1, args.total_timesteps+1):
    #     print(timestep)
    #     for subAgent in subAgents:
    #         subAgent.explorationStep(timestep, args.total_timesteps, args.SCAN_ID)
    #         filename = f'{subAgent.agent_id}_id.json'
    #         behavior_folder = os.path.join(folder_name, subAgent.behavior)
    #         with open(os.path.join(behavior_folder, filename), 'w') as f:
    #             json.dump(subAgent.observation_dict, f, indent = 4)


    


        
    

