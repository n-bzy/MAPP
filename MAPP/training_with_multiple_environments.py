from Experience_Replay_Buffer import ExperienceReplayBuffer
from Agent import Agent
import time
import numpy as np
from training_setup import hyperparameter_settings, create_env
import tensorflow as tf

num_environments, num_actions, ERP_size, num_training_samples, TIMESTEPS, EPISODES, epsilon, EPSILON_DECAY, MIN_EPSILON, MODEL_NAME, AGGREGATE_STATS_EVERY, MIN_REWARD = hyperparameter_settings()


# instantiate environments
environments = [create_env() for _ in range(num_environments)] 

# instantiate agent
Q_net = Agent(num_actions, num_environments, MODEL_NAME)
Q_net.update_delay_target_network()

ERPs = [ExperienceReplayBuffer(size = ERP_size) for _ in range(num_environments)]

for i in range(num_environments):
   ERPs[i].fill_up(environments[i])

print("ERPs filled with random samples")

reward_per_episode = []

for episode in range(EPISODES):

   # do a step function in every Environment, fill the ERP and collect a list of rewards (one for each in the list of environments)
   reward_of_episode = Q_net.fill_array_multiple_environments(environments, TIMESTEPS, ERPs, epsilon)
   # I took the average of all environments here that makes it easier to log it. I hope that is okay
   reward_per_episode.append(np.mean(reward_of_episode))

   Q_net.training_multiple_environments(num_training_samples, ERPs)

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

   # decay epsilon
   if epsilon > MIN_EPSILON:
      epsilon *= EPSILON_DECAY
      epsilon = max(MIN_EPSILON, epsilon)

   # we should consider printing only the average of the rewards
   print(f'done with epsiode {episode} with reward {reward_of_episode}') 