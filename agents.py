import numpy as np
import os

from helper import *
from methods import *

import torch.nn as nn
import torch.optim as optim
import torch
from collections import defaultdict
import networkx as nx
import json
from matterport_setup import *
import re
def load_nav_graph(scan_id):
    """Load connectivity graph for each scan."""

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    connectivity_dir = '/Users/bhrijpatel/Desktop/mp3d_results/connectivity/'
    with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan_id)) as f:
        G = nx.Graph()
        positions = {}
        data = json.load(f)
        for i, item in enumerate(data):
            if item['included']:
                for j, conn in enumerate(item['unobstructed']):
                    if conn and data[j]['included']:
                        positions[item['image_id']] = np.array([item['pose'][3],
                                item['pose'][7], item['pose'][11]]);
                        assert data[j]['unobstructed'][i], 'Graph should be undirected'
                        G.add_edge(item['image_id'], data[j]['image_id'], weight=distance(item, data[j]))
        #print(len(positions), i)
        nx.set_node_attributes(G, values=positions, name='position')
    return G

def viewpoint_from_object(detobjectdict, pred_object):
    
    max_count = 0
    sel_num = None

    # Iterate and count
    print('pred object', pred_object)
    for key, objects in detobjectdict.items():
        #print(objects)
        current_count = objects.count(pred_object.lower())
        if current_count > max_count:
            max_count = current_count
            sel_num = key
            
    return sel_num


def set_new_node(SCAN_ID, cur_node, sel_num):
    
    G = load_nav_graph(SCAN_ID.split('_')[0])
    newnode = determine_target_node(G, SCAN_ID.split('_')[0], cur_node, sel_num)

    if newnode is not None:
        cur_node = newnode
    else:
        adjnodes = list(G.neighbors(cur_node))
        np.random.shuffle(adjnodes)
        cur_node = np.random.choice(adjnodes)
    
    return newnode


def get_nearby_objects(SCAN_ID, cur_node, method):
    
    detobjectdict = defaultdict(list)

    #detector_pred_template = '/Users/bhrijpatel/Desktop/mp3d_results/detected_objs/{}/{}_{}.txt'
    detector_pred_template = '../glip_detections/{}/{}.txt'

    file_path = detector_pred_template.format(SCAN_ID.split('_')[0], cur_node)

    # Read the file
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            # Split the line by commas and strip spaces, then filter out empty strings
            # items = [item.strip() for item in line.split(',') if item.strip()]

            words = re.findall(r'\b\w+\b', line)

            items_list = [word for word in words if not any(char.isdigit() or char == ':' for char in word)]

            items_to_remove = ['wall','clutter', 'ceiling', 'box', 'door', 'column', 'bag', 'stairs', 'beam', 'floor']

            items = [item.lower() for item in items_list if item.lower() not in items_to_remove]
            detobjectdict[i+1].extend(items)
    # with open(file_path, 'r') as file:
    # # Read all lines into a list
    #     #item_lines = file.readlines()
    #     item_lines = file.read()
    # If you want to create a list of unique values (flattened from all lists), you can do:
    unique_items = set(item for sublist in detobjectdict.values() for item in sublist)
    valueset = list(unique_items)

    
    return detobjectdict, valueset


def get_image_matrix(node_data, selected_image_id):
    """
    Fetches the transformation matrix for the selected image id.
    Assumes node_data is a dict with keys as image ids (1-4) and values containing 'matrix'.
    """
    for key, value in node_data.items():
        #print(key, selected_image_id)
        if key.endswith(f'_d0_{selected_image_id}.png') or key.endswith(f'_i0_{selected_image_id}.jpg'):
            return value['matrix']
    return None

def parse_config_file(scan_id, node_id):
    scan_id = scan_id.split('_')[0]
    config_path = f"/Users/bhrijpatel/Desktop/Mp3D_House_Data/{scan_id}/undistorted_camera_parameters/{scan_id}.conf"
    node_data = {}
    with open(config_path, 'r') as file:
        for line in file:
            if line.startswith('scan') and node_id in line:
                parts = line.split()
                depth_image, color_image = parts[1], parts[2]
                matrix_values = np.array(parts[3:19], dtype=float).reshape(4, 4)
                node_data[depth_image] = {'color_image': color_image, 'matrix': matrix_values}
    return node_data

def get_forward_direction(matrix):
    # Extract the third column to get the forward direction
    forward_vector = -matrix[:3, 2]
    return forward_vector / np.linalg.norm(forward_vector)

def determine_target_node(G, scan_id, node_id, selected_image_id):
    # Parse the config file to get data for the node
    
    scan_id = scan_id.split('_')[0]
    node_data = parse_config_file(scan_id, node_id)
    
    # Fetch the transformation matrix for the selected image id
    selected_matrix = get_image_matrix(node_data, selected_image_id)
    if selected_matrix is None:
        return None
    
    forward_direction = get_forward_direction(selected_matrix)
    
    # Get the current node's position from the graph
    current_position = np.array(G.nodes[node_id]['position'])
    closest_adjacent_node_name = None
    highest_dot_product = -np.inf
    
    # Iterate through the neighbors of the current node
    for neighbor in G.neighbors(node_id):
        neighbor_position = np.array(G.nodes[neighbor]['position'])
        direction_to_neighbor = neighbor_position - current_position
        direction_to_neighbor_normalized = direction_to_neighbor / np.linalg.norm(direction_to_neighbor)
        
        # Calculate the dot product
        dot_product = np.dot(forward_direction, direction_to_neighbor_normalized)
        
        if dot_product > highest_dot_product:
            highest_dot_product = dot_product
            closest_adjacent_node_name = neighbor
    
    return closest_adjacent_node_name


class CentralClassifier(nn.Module):
    def __init__(self):
        super(CentralClassifier, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x




class subAgent:
    def __init__(self, agent_id, model, behavior, init_node):
        self.agent_id = agent_id
       
        self.model = model

        self.behavior = behavior

        self.memory_network =  AdvancedDynamicMemoryNetwork() 
        self.chosen_memory = ""

        self.node = init_node

        self.observation_dict = defaultdict(list)

    def setSystemPrompt(self, sys_prompt):
        self.sys_prompt = sys_prompt

 
    def explorationStep(self, timestep, total_timesteps, SCAN_ID):
        # pass

    
            #obj, subAgent.memory_network = LGX(timestep, total_timesteps, memory_network = subAgent.memory_network)[:2]
            
        detobjectdict, valueset = get_nearby_objects(SCAN_ID, self.node, method="YOLO")
            
        ## GETTING OBSERVATIONS
                        
        portable_objects = return_pobjs_in_node(timestep, self.node, scan = SCAN_ID.split('_')[0], seed = 2, episode = 1, movement = "routine")
        
        valueset = list(set(valueset + portable_objects))
        #self.observation_dict[timestep] = {'node': self.node, 'items': valueset}

        # print(self.observation_dict)
        # return 0 
        obj, _, self.chosen_memory = LGX(timestep, total_timesteps, objlist = valueset, chosen_memory = self.chosen_memory, behavior = self.behavior)


        self.observation_dict[timestep] = {'node': self.node, 'items': valueset, 'pred_object': obj}

        #print('lgx done')
        if obj != 'stay':
            sel_num = viewpoint_from_object(detobjectdict, obj)
            if sel_num is not None:
                #print('viepoint num selected')
                #print('selection_num', sel_num)
                self.node = set_new_node(SCAN_ID, self.node, sel_num)
        #print('new node set')




        
    

if __name__ == "__main__":
 
    central = centralAgent()


    agent1 = subAgent(agent_id=1, task="Task A", model="Model_X")
    agent2 = subAgent(agent_id=2, task="Task B", model="Model_Y")




