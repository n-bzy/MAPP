import gymnasium as gym
from Experience_Replay_Buffer import ExperienceReplayBuffer
from Agent import Agent
import time
import numpy as np


num_environments = 32 # 32 takes a long time for even one episode. Maybe we should lower that
num_actions = 6 #env.action_space shouldn't we then also set use_full_action_space=False ?
m = 50 # amount of training samples
# timesteps need to be large enough otherwise the agent don't have enough time to play a full game. The for-loop should be stopped by the done flag before this runs out
timesteps = 5000 # amount of samples to fill ERP with after the its filled up once
AGGREGATE_STATS_EVERY = 50 # get and safe stats every n episodes
MIN_REWARD = 0 # safe model only when the lowest reward of model over the last n episodes reaches a threshold
EPISODES = 20_000
ERP_size = 100
MODEL_NAME = "SinglePong" # used for saving and logging

# instantiate environments
env_name = 'ALE/Pong-v5' 
environments = [gym.make(env_name) for _ in range(num_environments)] #render_mode = 'human')

# instantiate q_network
Q_net = Agent(num_actions, num_environments, MODEL_NAME)
Q_net.update_delay_target_network()

ERP =  ExperienceReplayBuffer(size = ERP_size)

reward_per_episode = []

for episode in range(EPISODES):

   # do a step function in every Environment, fill the ERP and collect a list of rewards (one for each in the list of environments)
   reward_of_episode = Q_net.fill(environments, timesteps, ERP)

   for amount_of_samples in range(m):
      sample = ERP.sample()
      q_target = Q_net.q_target(sample) 
      observations = [sample[0] for sample in sample]
      Q_net.network.train(observations, q_target)

   # should we update this every episode?
   Q_net.update_delay_target_network()

   # I took the average of all environments here that makes it easier to log it. I hope that is okay
   reward_per_episode.append(np.mean(reward_of_episode))

   # every n episodes this safes the average and min / max rewards of these episodes to the tensorboard
   if not episode % AGGREGATE_STATS_EVERY or episode == 1:
      average_reward = sum(reward_per_episode[-AGGREGATE_STATS_EVERY:]) / len(reward_per_episode[-AGGREGATE_STATS_EVERY:])
      min_reward = min(reward_per_episode[-AGGREGATE_STATS_EVERY:])
      max_reward = max(reward_per_episode[-AGGREGATE_STATS_EVERY:])
      Q_net.tensorboard.update_stats(rewards_avg=average_reward, reward_min=min_reward, reward_max=max_reward)

      # Save model, but only when min reward is greater or equal a set value
      if min_reward >= MIN_REWARD:
         Q_net.model.save(
            f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

   # we should consider printing only the average of the rewards
   print(f'done with epsiode {episode} with reward {reward_of_episode}') 