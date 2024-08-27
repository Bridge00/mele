# Functions for various navigation approaches
from helper import *
import torch
import numpy as np
import os
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3 import PPO
from scenario_utils import *

def LGX(timestep, total_timesteps, objlist = None,  chosen_memory = None, target_obj = None, memory_network = None, use_memory = True, behavior = "aggressive"):


    
    objlist = list(set(objlist))
    #print("Objects found by YOLO are ", objlist)

    if use_memory and memory_network is not None:

        # Updating memory with observations
        memory_network.update_memory(objlist, timestep)

        # Query the memory for objects similar to "smartphone" with a similarity threshold
        queried_memory = memory_network.query_memory(target_obj, 0.5)

        #print("Queried Memory Based on Similarity to {}:".format(target_obj))
        
        max_sim = 0
        chosen_memory = ""

        for entry in queried_memory:
            # print(f"Timestep: {entry['timestep']}, Object: {entry['object']}, Similarity: {round(entry['similarity'], 2)}")
            if round(entry['similarity'], 2) >= max_sim:
                chosen_memory = "Timestep: {0}, Object: {1}, Similarity: {2}".format(entry['timestep'], entry['object'], round(entry['similarity'], 2))
                max_sim = round(entry['similarity'], 2)




    elif use_memory is True and memory_network is None:

        chosen_memory += "Timestep: {0}, Objects Seen: {1} \n".format(timestep, objlist)
        sys_prompt = exploration_sys_prompts[behavior] 
        sys_prompt = f' Here are a list of objects I have seen so far. {chosen_memory}'


		
    else:

        sys_prompt = "I am an embodied agent trying to find {0}.".format(target_obj)
    
    user_prompt = "I see objects {0} around me? I can either go towards an object or choose action \"stay\"\n \
    Reply using ONE WORD, with EITHER the object name from the list OR \"stay\". Do NOT SAY ANYTHING ELSE.".format(objlist)



    obj = gpt_run(sys_prompt, user_prompt, model="gpt-4-turbo")

    if obj.lower() not in objlist:
        obj = 'stay'
    # print("Predicted object is ", obj)

    return obj.lower(), memory_network, chosen_memory







