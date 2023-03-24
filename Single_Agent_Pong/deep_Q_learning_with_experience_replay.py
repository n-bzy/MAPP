from Experience_Replay_Buffer import ExperienceReplayBuffer
from Agent import Agent
import time
from training_setup import hyperparameter_settings, create_env
import numpy as np


# instantiate environment
env = create_env()

# set Hyperparameters
num_actions, ERP_size, EPISODES, epsilon, MODEL_NAME, AGGREGATE_STATS_EVERY, MIN_REWARD, UPDATE_TARGET_EVERY = hyperparameter_settings()

#instantiate and fill ERP
ERP =  ExperienceReplayBuffer(size = ERP_size)
ERP.fill(env)
print(f"ERP filled with {ERP.size} random samples")

# instantiate q_network
Q_net = Agent(env, ERP, MODEL_NAME)
Q_net.update_delay_target_network()

reward_per_episode = []


for episode in range(EPISODES):

    observation, _ = env.reset()
    observation = ERP.normalizing(observation)
    terminated, truncated = False, False
    Q_net.reward_of_game = 0

    # collect experiences and train the Agent
    while truncated == False and terminated == False:
        observation, terminated, truncated = Q_net.play(observation)
        Q_net.training()

    # logging 
    reward_per_episode.append(Q_net.reward_of_game)

    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(reward_per_episode[-AGGREGATE_STATS_EVERY:]) / len(reward_per_episode[-AGGREGATE_STATS_EVERY:])
        min_reward = min(reward_per_episode[-AGGREGATE_STATS_EVERY:])
        max_reward = max(reward_per_episode[-AGGREGATE_STATS_EVERY:])
        Q_net.tensorboard.update_stats(rewards_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                        epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            Q_net.network.save(
            f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # target network gets update every n episodes
    if not episode % UPDATE_TARGET_EVERY:
        Q_net.update_delay_target_network()

    print(f'done with epsiode {episode} with reward {Q_net.reward_of_game} (epsilon = {Q_net.epsilon})')

    