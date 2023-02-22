from pettingzoo.atari import pong_v3 
import numpy as np


# IGNORE the following four comments
# API - application programming interface: 
# AECEnv steps agents one at a time vs. parallel_env allowing for multiple agents; we can wrap AEC to parallel 


# [1] https://pettingzoo.farama.org/environments/atari/
# we will use supersuit sometime as atari is dterministic [1]


# select the environment, num_players \in {2,4}, render_mode "human" enables the visualization, also possible to get an array instead
env = pong_v3.env(num_players=2, render_mode = "human", max_cycles = 125000)

#env needs to be reseted
env.reset()

# render means displaying the env aka an image of pong
env.render()

# step is an action taken. Here 125000 actions are choosen 
# as we have 6 actions in pong, we choose randomly from {0, 1, 2, 3, 4, 5}
for i in range(125000):
    env.step(np.random.randint(0, 6))

