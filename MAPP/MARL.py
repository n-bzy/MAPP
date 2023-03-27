from pettingzoo.atari import pong_v3
import numpy as np
import tensorflow as tf
from Experience_Replay_Buffer import ExperienceReplayBuffer
from MARL_Agent import MARL_Agent
import training_setup
import time
import datetime

# load hyperparameters
num_actions, ERP_size, num_training_samples, EPISODES, epsilon, EPSILON_DECAY, MIN_EPSILON, MODEL_NAME, UPDATE_TARGET_EVERY, AGGREGATE_STATS_EVERY, MIN_REWARD = training_setup.hyperparameter_settings(AGGREGATE_STATS_EVERY=5)
print("hyperparameters loaded")

# create summary writer for logging
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_path = f"logs/{MODEL_NAME}/{current_time}"
summary_writer = tf.summary.create_file_writer(train_log_path)


env = training_setup.create_env()
print("environment created")

# instantiate and fill ERP
ERP = ExperienceReplayBuffer(size=ERP_size)
ERP.fill(env)
print(f"ERP filled with {ERP.size} random samples")

# instantiate q_network
Q_net = MARL_Agent(env, ERP, num_actions, MODEL_NAME, epsilon=epsilon, min_epsilon=MIN_EPSILON, epsilon_decay_value=EPSILON_DECAY)
print("Qnet created")

# store rewards
positive_reward_agent_one = []
negative_reward_agent_one = []
positive_reward_agent_two = []
negative_reward_agent_two = []
loss = []

print("starting training loop")
# training loop
for episode in range(EPISODES):

    env.reset()
    # reward is split up into the two agent and into positive and negative rewards separately
    # the first entry in each sublist is the positive rewards an agent got and the second entry the negative rewards
    episode_reward = [[0, 0], [0, 0]]  # to keep track of the reward of both agents in a single episode
    agent_index = 0

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

            # store new values in ERP
            ERP.experience_replay_buffer[ERP.index] = (observation, action, tf.cast(reward, tf.float32), next_observation)
            ERP.set_index()

            # decay Epsilon
            Q_net.epsilon_decay()

            # store rewards for later logging
            if reward > 0:
                episode_reward[agent_index][0] = episode_reward[agent_index][0] + reward
            elif reward < 0:
                episode_reward[agent_index][1] = episode_reward[agent_index][1] + reward
            agent_index = (agent_index + 1) % 2

            # do a training step
            metrics = Q_net.training()

    # the target network only gets updated every few episodes
    if not episode % UPDATE_TARGET_EVERY:
        Q_net.update_delay_target_network()

    # collect the rewards for each agent separately
    # as a zero-sum game the sum of both will be zero
    positive_reward_agent_one.append(episode_reward[0][0])
    negative_reward_agent_one.append(episode_reward[0][1])
    positive_reward_agent_two.append(episode_reward[1][0])
    negative_reward_agent_two.append(episode_reward[1][1])

    for (key, value) in metrics.items():
        if key == 'loss':
            loss.append(value)

    epsilon = Q_net.epsilon

    # every n episodes this safes the average and min / max rewards of these episodes to the tensorboard
    # positive and negative rewards are logged separately
    # this avoids the issue that if both agents are being bad a defense the reward will cancel out to be close to 0
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:

        agent_one_average_positive_reward = sum(positive_reward_agent_one[-AGGREGATE_STATS_EVERY:]) / len(
            positive_reward_agent_one[-AGGREGATE_STATS_EVERY:])
        agent_one_average_negative_reward = sum(negative_reward_agent_one[-AGGREGATE_STATS_EVERY:]) / len(
            negative_reward_agent_one[-AGGREGATE_STATS_EVERY:])

        agent_two_average_positive_reward = sum(positive_reward_agent_two[-AGGREGATE_STATS_EVERY:]) / len(
            positive_reward_agent_two[-AGGREGATE_STATS_EVERY:])
        agent_two_average_negative_reward = sum(negative_reward_agent_two[-AGGREGATE_STATS_EVERY:]) / len(
            negative_reward_agent_two[-AGGREGATE_STATS_EVERY:])

        agent_one_min_positive_reward = min(positive_reward_agent_one[-AGGREGATE_STATS_EVERY:])
        agent_one_min_negative_reward = min(negative_reward_agent_one[-AGGREGATE_STATS_EVERY:])

        agent_two_min_positive_reward = min(positive_reward_agent_two[-AGGREGATE_STATS_EVERY:])
        agent_two_min_negative_reward = min(negative_reward_agent_two[-AGGREGATE_STATS_EVERY:])

        agent_one_max_positive_reward = max(positive_reward_agent_one[-AGGREGATE_STATS_EVERY:])
        agent_one_max_negative_reward = max(negative_reward_agent_one[-AGGREGATE_STATS_EVERY:])

        agent_two_max_positive_reward = max(positive_reward_agent_two[-AGGREGATE_STATS_EVERY:])
        agent_two_max_negative_reward = max(negative_reward_agent_two[-AGGREGATE_STATS_EVERY:])

        average_loss = sum(loss[-AGGREGATE_STATS_EVERY:]) / len(loss[-AGGREGATE_STATS_EVERY:])

        with summary_writer.as_default():
            tf.summary.scalar(f"rewards_avg_pos_1", agent_one_average_positive_reward, step=episode)
            tf.summary.scalar(f"rewards_avg_neg_1", agent_one_average_negative_reward, step=episode)
            tf.summary.scalar(f"rewards_avg_pos_2", agent_two_average_positive_reward, step=episode)
            tf.summary.scalar(f"rewards_avg_neg_2", agent_two_average_negative_reward, step=episode)
            tf.summary.scalar(f"rewards_max_pos_1", agent_one_max_positive_reward, step=episode)
            tf.summary.scalar(f"rewards_max_neg_1", agent_one_max_negative_reward, step=episode)
            tf.summary.scalar(f"rewards_max_pos_2", agent_two_max_positive_reward, step=episode)
            tf.summary.scalar(f"rewards_max_neg_2", agent_two_max_negative_reward, step=episode)
            tf.summary.scalar(f"rewards_min_pos_1", agent_one_min_positive_reward, step=episode)
            tf.summary.scalar(f"rewards_min_neg_1", agent_one_min_negative_reward, step=episode)
            tf.summary.scalar(f"rewards_min_pos_2", agent_two_min_positive_reward, step=episode)
            tf.summary.scalar(f"rewards_min_neg_2", agent_two_min_negative_reward, step=episode)
            tf.summary.scalar(f"loss_avg", average_loss, step=episode)
            tf.summary.scalar(f"epsilon", epsilon, step=episode)


        # Save model, but only when min reward is greater or equal a set value
        if max(agent_one_min_positive_reward, agent_two_min_positive_reward) >= MIN_REWARD:
            Q_net.network.save(
                f'models/{MODEL_NAME}__{agent_one_average_positive_reward:_>7.2f}pos_avg_{agent_one_average_negative_reward:_>7.2f}neg_avg__{int(time.time())}.model')


    print(f'done with epsiode {episode} with reward {episode_reward} and loss {loss[episode]} and epsilon {epsilon}')

#%%
