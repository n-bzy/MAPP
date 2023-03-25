from pettingzoo.atari import pong_v3
import numpy as np
import tensorflow as tf
from Experience_Replay_Buffer import ExperienceReplayBuffer
from MARL_Agent import MARL_Agent
import time

# Hyperparameters
num_actions = 6  # env.action_space shouldn't we then also set use_full_action_space=False ?
NUM_TRAINING_SAMPLES = 1000  # amount of training samples
AGGREGATE_STATS_EVERY = 50  # get and safe stats every n episodes
UPDATE_TARGET_EVERY = 5  # update the target network every n episodes
MIN_REWARD = 0  # safe model only when the lowest reward of model over the last n episodes reaches a threshold
EPISODES = 20_000
ERP_size = 1_000
MODEL_NAME = "MultiPong"  # used for saving and logging

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

# instantiate environment
#env = pong_v3.env(num_players=2, render_mode="human", max_cycles=125000)
env = pong_v3.env(num_players=2)

# instantiate and fill ERP
ERP = ExperienceReplayBuffer(size=ERP_size)
ERP.fill(env)
print(f"ERP filled with {ERP.size} random samples")

# instantiate q_network
Q_net = MARL_Agent(env, ERP, num_actions, MODEL_NAME, epsilon=epsilon, min_epsilon=MIN_EPSILON, epsilon_decay_value=EPSILON_DECAY)

# store rewards
reward_per_episode = []

# training loop
for episode in range(EPISODES):

    env.reset()
    episode_reward = [0, 0]  # to keep track of the reward of both agents in a single episode
    a = 0

    # agent gets removed from env.agents when it is flagged as done
    while env.agents:
        for agent in env.agent_iter():

            observation, reward, termination, truncation, info = env.last()
            observation = ERP.normalizing(observation)
            action = Q_net.epsilon_greedy_sampling(observation)
            env.step(action)

            #FIXME: last() only returns the current observation, and step() returns nothing.
            # calling last() again after doing a step seems to be the only way of getting the next observation
            next_observation, _, _, _, _ = env.last()
            next_observation = ERP.normalizing(next_observation)

            ERP.experience_replay_buffer[ERP.index] = (observation, action, tf.cast(reward, tf.float32), next_observation)
            ERP.set_index()

            Q_net.epsilon_decay()

            episode_reward[a % 2] = episode_reward[a % 2] + reward  # only works with two agents
            a += 1


    # skip the training until the ERP is filled with enough values
    if ERP.size >= MINIMUM_ERP_SIZE:
        Q_net.training()

    # the target network only gets updated every few episodes
    if not episode % UPDATE_TARGET_EVERY:
        Q_net.update_delay_target_network()

    # In the MARL case we should probably safe the rewards for every agent separately
    # so we need to update this at some point
    # with parameter sharing, it shouldn't make a difference which agent got which reward
    # might still be good to catch irregularities
    reward_per_episode.append(np.mean(episode_reward))

    # every n episodes this safes the average and min / max rewards of these episodes to the tensorboard
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(reward_per_episode[-AGGREGATE_STATS_EVERY:]) / len(
            reward_per_episode[-AGGREGATE_STATS_EVERY:])
        min_reward = min(reward_per_episode[-AGGREGATE_STATS_EVERY:])
        max_reward = max(reward_per_episode[-AGGREGATE_STATS_EVERY:])
        Q_net.tensorboard.update_stats(rewards_avg=average_reward, reward_min=min_reward, reward_max=max_reward)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            Q_net.model.save(
                f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # we should consider printing only the average of the rewards
    print(f'done with epsiode {episode} with reward {episode_reward}')

#%%
