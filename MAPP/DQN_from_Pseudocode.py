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

import gymnasium as gym
from Experience_Replay_Buffer import ExperienceReplayBuffer
from Agent import Agent
import time
import gymnasium.wrappers as gw
import numpy as np
from training_setup import hyperparameter_settings, create_env



# instantiate environment
env = create_env()

num_environments, num_actions, ERP_size, num_training_samples, TIMESTEPS, EPISODES, epsilon, EPSILON_DECAY, MIN_EPSILON, MODEL_NAME, AGGREGATE_STATS_EVERY, MIN_REWARD = hyperparameter_settings(epsilon = 0.1)

# instantiate q_network
Q_net = Agent(num_actions, 1, MODEL_NAME)
Q_net.update_delay_target_network()

ERP =  ExperienceReplayBuffer(size = 1000) 

reward_per_episode = []

ERP.fill_up(env)

for episode in range(EPISODES):
   reward_of_episode = Q_net.fill_array(environment = env, timesteps = TIMESTEPS, ERP = ERP, epsilon = epsilon)

   Q_net.training(num_training_samples, ERP)

   reward_per_episode.append(reward_of_episode)

   if not episode % AGGREGATE_STATS_EVERY or episode == 1:
      average_reward = sum(reward_per_episode[-AGGREGATE_STATS_EVERY:]) / len(reward_per_episode[-AGGREGATE_STATS_EVERY:])
      min_reward = min(reward_per_episode[-AGGREGATE_STATS_EVERY:])
      max_reward = max(reward_per_episode[-AGGREGATE_STATS_EVERY:])
      Q_net.tensorboard.update_stats(rewards_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                     epsilon=epsilon)

      # Save model, but only when min reward is greater or equal a set value
      if min_reward >= MIN_REWARD:
         Q_net.model.save(
            f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

   print(f'done with epsiode {episode} with reward {reward_of_episode}')










