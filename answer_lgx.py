import numpy as np
from helper import *
import torch
import numpy as np
import os
from scenario_utils import *
import argparse
import csv
import pandas as pd
import json
import pickle
import random
from agents import get_nearby_objects
from collections import defaultdict
def create_question( room, object, timestep = -1):
    # if timestep == -1:
    #     timestep = random.randint(1, 500)
    #return f"At timestep {timestep}, in the {room}, was there a {object}? Respond with only YES or NO. No other answer and just give a one-word response. Remember, half of the answers will be Yes and half with be No. So please vary your responses evenly"
    #return f"Do you think there is a {object} in the {room} at timestep {timestep}? Respond with only YES or NO. No other answer and just give a one-word response. Remember, half of the answers will be Yes and half with be No. So please vary your responses evenly. Even if you did not see it in your observation list, the answer could still be yes."
    #return f"Do you think there is a {object} in the {room} at timestep {timestep}? Respond with only YES or NO. If NO, explain your answer" 
    #return f"Do you think there is a {object} in the {room}? If the answer is not in your observations, use common-sense reasoning about the object and room. Respond with YES or NO. If NO, explain your reasoning." 
    return f"Do you think there is a {object} in the {room}? Use both common-sense reasoning about the object and room and the observation list given. Even if the relevant information is not in your observations, the answer can still be YES. Respond with YES or NO." 
    #return f"Do you think there is a {object} in the {room}? Even if the relevant information is not in your observations, the answer can still be YES. Respond with YES or NO." 


# def obs_to_string(timestep, observation_file, dict_):
#     obs_list = []
#     for i in range(timestep+1):
        

#     pass

#HORIZON = 50
def get_response(question,  obs_list):

    
    
    sys_prompt = 'You are an embodied agent that has explored a house.'

    sys_prompt += ' I prefer definite answers to questions which might require to explore. The observations are: \n'
    sys_prompt += str(list(obs_list))

    response = gpt_run(sys_prompt, question, model="gpt-4-turbo")



    return response

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multi-LLM answers')

    parser.add_argument('--num_questions', '-q', type=int, default= 100, metavar='N',
                        help='number of questions')

    parser.add_argument('--behavior', '-b', type=str, default= 'aggressive', metavar='N',
                        help='exploration behavior')
    
    parser.add_argument('--agent_id', '-a', type=int, default= 0, metavar='N',
                    help='agent id') 
    
    parser.add_argument('--SCAN_ID', '-scan', type=str, default= 'zsNo4HB9uLZ', metavar='N',
                    help='scan_ID') 
    
    parser.add_argument('--seed', '-s', type=int, default= 0, metavar='N',
                    help='routine seed') 
    
    parser.add_argument('--answer_behavior', type=str, default= 'helpful', metavar='N',
                    help='') 
    parser.add_argument('--method', type=str, default= 'glip', metavar='N',
                    help='') 
    
    parser.add_argument('--full_mem','-fm', type=int, default= 0, metavar='N',
                    help='') 
    parser.add_argument('--reload_questions_from_source','-r', type=int, default= 1, metavar='N',
                    help='') 

    args = parser.parse_args() 

    if False:
        answer_folder = 'answers_fixed_exp_balanced'
        os.makedirs(f'{answer_folder}/{args.method}/{args.SCAN_ID}/independent_answers/', exist_ok=True)
        #answer_file = f'{answer_folder}/{args.method}/{args.SCAN_ID}/answers_{args.seed}_with_obs_full_mem_{args.full_mem}.csv'

        if True or not os.path.exists(answer_file) or args.reload_questions_from_source:

            df_0 = pd.read_csv(f'questions/{args.method}/{args.SCAN_ID}/no_queries.csv')
            df_1 = pd.read_csv(f'questions/{args.method}/{args.SCAN_ID}/yes_queries.csv')
            data = pd.concat([df_0, df_1])
        else:
            data = pd.read_csv(answer_file)

        questions = []
        for index, row in data.iterrows():
            #print(row['Timestep'])
            #print(row['Object'])
            #questions.append(create_question(row['Timestep'], row['Room'], row['Object']))
            questions.append(create_question(row['Room'], row['Object']))

        with open('scan_to_node_to_room.pkl', 'rb') as f:
            SCAN_TO_NODE_TO_ROOM = pickle.load(f)

        #room_combinations = [['bathroom'], ['bedroom'], ['dining room'], ['kitchen'], ['dining room', 'kitchen'], ['bathroom', 'bedroom'], ['office', 'familyroom']]
        room_combinations =['bathroom', 'kitchen', 'dining room', 'bedroom', 'office', 'familyroom', 'living room']#[['bathroom'], ['bedroom'], ['dining room'], ['kitchen'], ['dining room', 'kitchen'], ['bathroom', 'bedroom'], ['office', 'familyroom']]
        for desired_rooms in room_combinations:
            room_items = defaultdict(list)

            for node in SCAN_TO_NODE_TO_ROOM[args.SCAN_ID]:
                room = SCAN_TO_NODE_TO_ROOM[args.SCAN_ID][node]
                if room in [desired_rooms]:
                    #print(node)
                    x = get_nearby_objects(args.SCAN_ID, node, args.method)[1]
                    #print(x)
                    room_items[room] += x
            for room in [desired_rooms]:
                room_items[room] = list(set(room_items[room]))
            mem = room_items.items()

            #print(room_items)
            
            answers = []
            for index, row in data.iterrows():
                print(index)
                answers.append(get_response(questions[index],  mem))
            
            data[f'Response'] = answers

            answer_file = f'{answer_folder}/{args.method}/{args.SCAN_ID}/independent_answers/{desired_rooms}.csv'
            data.to_csv(answer_file, index=False)
            
    else:
        os.makedirs(f'answers_fixed_exp_balanced/{args.method}/{args.SCAN_ID}/independent_answers/', exist_ok=True)

        answer_file = f'answers_fixed_exp_balanced/{args.method}/{args.SCAN_ID}/independent_answers/{args.agent_id}.csv'

        if True:
            df_0 = pd.read_csv(f'questions/{args.method}/{args.SCAN_ID}/no_queries.csv')
            df_1 = pd.read_csv(f'questions/{args.method}/{args.SCAN_ID}/yes_queries.csv')
            data = pd.concat([df_0, df_1])

        else:
            data = pd.read_csv(answer_file)

        questions = []
        for index, row in data.iterrows():
            #print(row['Timestep'])
            print(row['Object'])
            #questions.append(create_question(row['Timestep'], row['Room'], row['Object']))
            questions.append(create_question(row['Room'], row['Object']))

        with open(f'observation_dictionaries/{args.method}/{args.SCAN_ID}/{args.behavior}/{args.agent_id}_id.json', 'r') as file:
            dict_ = json.load(file)

        with open('scan_to_node_to_room.pkl', 'rb') as f:
            SCAN_TO_NODE_TO_ROOM = pickle.load(f)

        #obs_list = [[int(time), SCAN_TO_NODE_TO_ROOM[args.SCAN_ID][dict_[time]['node']], dict_[time]['items']] for time in dict_]
        obs_list = [[ SCAN_TO_NODE_TO_ROOM[args.SCAN_ID][dict_[time]['node']], dict_[time]['items']] for time in dict_]
        if not args.full_mem:
            room_items = defaultdict(list)

            for time in dict_:
                room_items[SCAN_TO_NODE_TO_ROOM[args.SCAN_ID][dict_[time]['node']]].extend(dict_[time]['items'])

            for room in room_items:
                room_items[room] = list(set(room_items[room]))

            objlist = room_items.items()

        else:

            objlist = []

            for time in dict_:
                objlist.append((SCAN_TO_NODE_TO_ROOM[args.SCAN_ID][dict_[time]['node']], dict_[time]['items']))


        answers = []

        for index, row in data.iterrows():
            print(index)
            answers.append(get_response(questions[index],  objlist))

        
        data[f'Response'] = answers#data.apply(lambda row: get_response(row['Timestep'], row['question'], args.behavior, args.agent_id, args.answer_behavior, obs_list))

        #data.reindex(columns=[col for col in data.columns if col != 'question'] + ['question'])

        data.to_csv(answer_file, index=False)
    
   
#response = gpt_run(sys_prompt, user_prompt, model="gpt-4")

