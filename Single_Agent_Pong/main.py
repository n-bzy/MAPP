from Experience_Replay_Buffer import ExperienceReplayBuffer
from Agent import Agent
import time
from training_setup import hyperparameter_settings, create_env
import tensorflow as tf
import datetime


# instantiate environment
env = create_env()

# set Hyperparameters
num_actions, ERP_size, EPISODES, epsilon, MODEL_NAME, AGGREGATE_STATS_EVERY, MIN_REWARD, UPDATE_TARGET_EVERY = hyperparameter_settings()

# create summary writer for logging
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_path = f"logs/{MODEL_NAME}/{current_time}"
summary_writer = tf.summary.create_file_writer(train_log_path)

#instantiate and fill ERP
ERP =  ExperienceReplayBuffer(size = ERP_size)
ERP.fill(env)
print(f"ERP filled with {ERP.size} random samples")

# instantiate q_network
Q_net = Agent(env, ERP, MODEL_NAME)
Q_net.update_delay_target_network()

reward_per_episode = []
loss = []


for episode in range(EPISODES):

    observation, _ = env.reset()
    observation = ERP.normalizing(observation)
    terminated, truncated = False, False
    Q_net.reward_of_game = 0

    # collect experiences and train the Agent
    while truncated == False and terminated == False:
        observation, terminated, truncated = Q_net.play(observation)
        metrics = Q_net.training()

        for (key, value) in metrics.items():
            if key == 'loss':
                loss.append(value)

    # logging 
    reward_per_episode.append(Q_net.reward_of_game)


    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(reward_per_episode[-AGGREGATE_STATS_EVERY:]) / len(reward_per_episode[-AGGREGATE_STATS_EVERY:])
        min_reward = min(reward_per_episode[-AGGREGATE_STATS_EVERY:])
        max_reward = max(reward_per_episode[-AGGREGATE_STATS_EVERY:])
        average_loss = sum(loss[-AGGREGATE_STATS_EVERY:]) / len(loss[-AGGREGATE_STATS_EVERY:])
        #Q_net.tensorboard.update_stats(rewards_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
        #                                epsilon=epsilon)

        with summary_writer.as_default():
            tf.summary.scalar(f"average_reward", average_reward, step=episode)
            tf.summary.scalar(f"min_reward", min_reward, step=episode)
            tf.summary.scalar(f"max_reward", max_reward, step=episode)
            tf.summary.scalar(f"loss", average_loss, step=episode)

        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            Q_net.network.save(
            f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # target network gets update every n episodes
    if not episode % UPDATE_TARGET_EVERY:
        Q_net.update_delay_target_network()

    print(f'done with epsiode {episode} with reward {Q_net.reward_of_game} (epsilon = {Q_net.epsilon})')

    
