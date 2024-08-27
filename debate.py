import numpy as np
from helper import *
import argparse
import csv
import pandas as pd
import os
import pickle
from collections import defaultdict
import random
from agents import get_nearby_objects
#from answer import create_question
from time import time

def debate(X_val, scan, rooms, trial, label_encoders, malicious):
    preds = []
    rounds_per_question = 2
    print(len(X_val))
    all_conversations = []
    
    for row_index, row in X_val.iterrows():
        start = time()
        conversation = []
        
        with open('scan_to_node_to_room.pkl', 'rb') as f:
            SCAN_TO_NODE_TO_ROOM = pickle.load(f)

        #room_combinations = [['bathroom'], ['bedroom'], ['dining room'], ['kitchen'], ['dining room', 'kitchen'], ['bathroom', 'bedroom'], ['office', 'familyroom']]
        #room_combinations = ['bathroom', 'kitchen', 'dining room', 'bedroom', 'office', 'familyroom', 'living room']#[['bathroom'], ['bedroom'], ['dining room'], ['kitchen'], ['dining room', 'kitchen'], ['bathroom', 'bedroom'], ['office', 'familyroom']]
        for conv_round in range(rounds_per_question):
            print('conv_round',conv_round)
            for index, room_agent in enumerate(rooms):
            
                room_items = defaultdict(list)

                for node in SCAN_TO_NODE_TO_ROOM[scan]:
                    room = SCAN_TO_NODE_TO_ROOM[scan][node]
                    if room in [room_agent]:
                        #print(node)
                        x = get_nearby_objects(scan, node, 'glip')[1]
                        #print(x)
                        room_items[room] += x
                for room in [room_agent]:
                    room_items[room] = list(set(room_items[room]))
                mem = room_items.items()

                object_string = label_encoders['Object'].inverse_transform([row['Object']])[0]
                room_string = label_encoders['Room'].inverse_transform([row['Room']])[0]

                #print(object_string, room_string)
                initial_answer = 'Yes' if row[f'{room_agent}_response'] else 'No'
                #print('initial_naswer', row[f'{room_agent}_response'], initial_answer)
                sys_prompt = f"You are an embodied agent in a house. Your id is {index}. Here are you observations from exploring the house: {str(list(mem))}. Your initial answer to whether or not there was a {object_string} in the {room_string} was {initial_answer.upper()}. Other agents may have different answers. Please debate with the other agents to come to a consensus."
                user_prompt = f" It is your turn to speak. Use your observation and conversation history to help. "


                if len(conversation) > 0:
                    sys_prompt += f" Here is the conversation history: {conversation}" 
                else:
                    sys_prompt += f" This is the beginning of the conversation."

                response = gpt_run(sys_prompt, user_prompt, model="gpt-4-turbo")

                conversation.append(str((f'Round {index + 1}/{rounds_per_question}', f'Agent {index}', response)))
        final_answers = [] 
        for index, room_agent in enumerate(rooms):

            sys_prompt = f"You are an embodied agent in a house. Your id is {index}. Here are you observations from exploring the house: {str(list(mem))}. Your initial answer to whether or not there was a {object_string} in the {room_string} was {initial_answer.upper()}. Other agents may have different answers. Please debate with the other agents to come to a consensus."
            sys_prompt += f" Here is the conversation history: {conversation}" 

            user_prompt = 'Please give your final "Yes/No" answer. Give a definite answer.'

            answer = gpt_run(sys_prompt, user_prompt, model="gpt-4-turbo")
            conversation.append(('Final', f'Agent {index}', answer))

            val = 0 if 'no' in answer.lower() else 1

            final_answers.append(val)
        total = np.sum(final_answers)
        prediction = int(total > len(final_answers)/2)
        #print(time() - start)
        preds.append(prediction)
        #print(preds)
        all_conversations.append(conversation)
        os.makedirs(f'classification/glip/{scan}/debate/', exist_ok=True)
        file_name = f'{rooms}_trial_{trial}_conversation.txt'
        if malicious:
            file_name = f'{rooms}_trial_{trial}_conversation_malicious_agent.txt'
        with open(f'classification/glip/{scan}/debate/{file_name}', 'w') as f:
            #pickle.dump(all_conversations, f)
            for c in all_conversations:
                for recording in c:
                    f.write(str(recording))
        
    return preds