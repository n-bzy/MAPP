'''
DQN PSEUDOCODE
state space high dim, action space finite

initialize: Qnet(s,a)
and an empty experience replay buffer experience_replay_buffer

of maximum size N
for each episode:
     s = initial state
     for each timestep t:
         sample action at
        obtain reward rtand next state st+1
        store {st,at,rt,st+1} in experience_replay_buffer
     for m
random samples {st,at,rt,st+1}∈ experience_replay_buffer
   (t is here different from the t above!)
         compute qtarget=yt=rt+γmaxat+1Q(st+1,at+1)
        train Qnet(s,a) via backpropagation minimizing (yt−Qnet(st,at))²
        st=st+1
'''

from DQN import DQN 
import numpy as np
import gymnasium as gym
import tensorflow as tf


num_actions = 6 
n = 50 # size of experience_replay_buffer: e.g. 2000
m = 30 # amount of samples 
gamma = 0.95 #discount fator

# instantiate environment
env_name = 'ALE/Pong-v5' 
env = gym.make(env_name)

# instantiate q_network
Q_net = DQN(num_actions)  

experience_replay_buffer = [] 

for episode in range(10):
   observation, _ = env.reset() # state = observation
   #env.render()
   for timestep in range(n):
      observation_t = observation # observation_t represents state_t 
      action =  np.argmax(Q_net(observation_t)) #this is greedy for now, should be epsilon-greedy
      observation, reward, terminated, truncated, info = env.step(action) 
      experience_replay_buffer.append([observation_t, action, reward, observation]) 
   for amount_of_samples in range(m):
      sample = experience_replay_buffer[np.random.randint(n)] 
      q_values = Q_net(sample[3])
      max_q_value = tf.math.top_k(q_values, k=1, sorted=True) 
      q_target = sample[2] + gamma * max_q_value.values.numpy() 
      Q_net.train(sample[0], q_target)
   print(f'done with epsiode {episode}')






