num_agents = 3
exploration_scenarios = [

    {

        "subordinate_agent_models": ['gpt-4-turbo'] * num_agents,

        "exploration_behaviors": ['aggressive'] * num_agents,

    },

    {

        "subordinate_agent_models": ['gpt-4'] * 3,

        "exploration_behaviors": ['aggressive']  + ['conservative']  + ['neutral'] ,

    },

    {

        "subordinate_agent_models": ['gpt-4'] * num_agents,

        "exploration_behaviors": ['aggressive'] * int(num_agents/3) + ['conservative'] * int(num_agents/3) + ['neutral'] * int(num_agents/3),

    },

    # {
    #     "central_agent_model": 'gpt-4',

    #     "subordinate_agent_models": ['gpt-4', 'gpt-4', 'gpt-4'],

    #     "subordinate_agent_behaviors": ['aggressive', 'suggestible', 'aggressive'],

    #     "intent": ['helpful', 'helpful', 'helpful']
    # },

    # {
    #     "central_agent_model": 'gpt-4',

    #     "subordinate_agent_models": ['llama', 'claude', 'gemini', 'gpt-4'],

    #     "subordinate_agent_behaviors": ['aggressive', 'suggestible', 'neutral'],

    #     "intent": ['malicious', 'helpful', 'helpful']

    # }
]


exploration_sys_prompts = {
    'aggressive':  "I am an embodied agent in a house. I want to aggressively explore as much as I can, and I want to find as many unique objects as I can. When I see a list of objects to go towards to, I should try to choose objects which I have not gone towards before.",
    'conservative': "I am an embodied agent in a house. While I want to find as many unique objects as I can, I want to conserve energy and stay conservative with my exploration approach.",
    'neutral': "I am an embodied agent in a house. I want to find as many unique objects as I can."
}


#raw observations + recommendation of subordinates. either/or or mix

#exploration phase + task phase
#agents will keep exploring with cetain behaviors for a certain number of timesteps
#then be asked a question
#then will communicate with each other and give recs to central
# or send raw observations to central
# central will have it's final decision
# Metrics: Success rate, number of communiations 
