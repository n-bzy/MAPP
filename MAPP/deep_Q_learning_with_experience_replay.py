from Experience_Replay_Buffer import ExperienceReplayBuffer
from Agent import Agent
import time
import numpy as np
from training_setup import hyperparameter_settings, create_env
import tensorflow as tf


# instantiate environment
env = create_env()

num_environments, num_actions, ERP_size, num_training_samples, TIMESTEPS, EPISODES, epsilon, EPSILON_DECAY, MIN_EPSILON, MODEL_NAME, AGGREGATE_STATS_EVERY, MIN_REWARD = hyperparameter_settings()

# instantiate q_network
Q_net = Agent(num_actions, 1, MODEL_NAME)
Q_net.update_delay_target_network()

ERP =  ExperienceReplayBuffer(size = TIMESTEPS) 

reward_per_episode = []

ERP.fill_up(env)
ERP.experience_replay_buffer = ERP.preprocessing()

start = time.time()
for episode in range(EPISODES):
    reward_of_episode = Q_net.fill_array(environment = env, timesteps = TIMESTEPS, ERP = ERP, epsilon = epsilon)

    data = ERP.preprocessing()
    ERP.experience_replay_buffer.concatenate(data)

    def training(Q_net, data):
        for batch in data:
            q_target = q_target_array(Q_net, batch)
            obs, action, _, _ = batch
            Q_net.network.train(obs, action, q_target)
            
        Q_net.update_delay_target_network()

    def q_target_array(Q_net, batch, discount_factor = 0.95):
        """
        Calculates Q-target (expected reward) with Delay-Target-Network.

        Parameters: 
            sample (list): list of lists each filled with [observation_t, action, reward, observation] 
            discount_factor (float): to set the influence of future rewards 

        Returns: 
            q_target (float): expected reward from "optimal" action 
        """
        observation, action, reward, next_observation = batch

        q_values = Q_net.delay_target_network(next_observation) #no time_distributed leaves shape = (4,6), mit time_distirbuted shape = (1,6)
        max_q_value = tf.math.reduce_max(q_values, axis = 1) #returns maximum for each batch
        q_target = reward + discount_factor * max_q_value
        return q_target

    training(Q_net, data)

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

    # decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    print(f'done with epsiode {episode} with reward {reward_of_episode}')
end = time.time()