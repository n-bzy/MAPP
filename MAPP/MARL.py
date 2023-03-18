from pettingzoo.atari import pong_v3
import numpy as np
from Experience_Replay_Buffer import ExperienceReplayBuffer
from MARL_Agent import MARL_Agent
import time

num_actions = 6  # env.action_space shouldn't we then also set use_full_action_space=False ?
NUM_TRAINING_SAMPLES = 1000  # amount of training samples
AGGREGATE_STATS_EVERY = 50  # get and safe stats every n episodes
UPDATE_TARGET_EVERY = 5  # update the target network every n episodes
MIN_REWARD = 0  # safe model only when the lowest reward of model over the last n episodes reaches a threshold
EPISODES = 20_000
ERP_size = 20_000
MINIMUM_ERP_SIZE = ERP_size / 2  # just to allow for a variable ERP_size
MODEL_NAME = "MultiPong"  # used for saving and logging

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

env = pong_v3.env(num_players=2, render_mode="human", max_cycles=125000)
Q_net = MARL_Agent(num_actions, MODEL_NAME)

ERP = ExperienceReplayBuffer(size=ERP_size)

reward_per_episode = []

ERP.fill_up(env)
reward_of_episode = np.sum(ERP.reward)
ERP.experience_replay_buffer = ERP.preprocessing()

for episode in range(EPISODES):

    env.reset()
    episode_reward = [0, 0]  # to keep track of the reward of both agents in a single episode
    a = 0

    # agent gets removed from env.agents when it is flagged as done
    while env.agents:
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            # FIXME: in the single agent case it is clear where we get the current and the next observation from but what about here?
            ERP.next_observation[ERP.index], ERP.reward[ERP.index], = observation, reward
            action = Q_net.epsilon_greedy_sampling(observation, epsilon)
            ERP.action[ERP.index] = action
            env.step(action)
            episode_reward[a % 2] = episode_reward[a % 2] + reward  # only works with two agents

    data = ERP.preprocessing()
    ERP.experience_replay_buffer.concatenate(data)

    # skip the training until the ERP is filled with enough values
    if ERP.size >= MINIMUM_ERP_SIZE:
        Q_net.training(data)

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

    # decay epsilon only when ERP is filled enough
    if ERP_size >= MINIMUM_ERP_SIZE:
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    # we should consider printing only the average of the rewards
    print(f'done with epsiode {episode} with reward {episode_reward}')
