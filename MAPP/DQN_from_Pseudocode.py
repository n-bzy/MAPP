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



# instantiate environment
env_name = 'ALE/Pong-v5' 
env = gym.make(env_name) #render_mode = 'human')

num_actions = 6 #env.action_space
m = 500 # amount of training samples
# timesteps need to be large enough otherwise the agent don't have enough time to play a full game. The for-loop should be stopped by the done flag before this runs out
timesteps = 5000 # amount of samples to fill ERP with after the its filled up once
AGGREGATE_STATS_EVERY = 50 # get and safe stats every 50 episodes
MIN_REWARD = 0 # safe model only when the lowest reward of model over the last n episodes reaches a threshold
EPISODES = 20_000

# instantiate q_network
Q_net = Agent(num_actions)
Q_net.update_delay_target_network()

experience_replay_buffer =  ExperienceReplayBuffer(size = 1000) 

reward_per_episode = []

for episode in range(EPISODES):

   reward_of_episode = experience_replay_buffer.fill(environment = env, Q_net = Q_net, timesteps = timesteps)
   #reward = sum([reward[2] for reward in experience_replay_buffer.experience_replay_buffer]) #der summiert gerade über alle anstatt nur den Neuen

   for amount_of_samples in range(m):
      sample = experience_replay_buffer.sample()
      q_target = Q_net.q_target(sample)
      Q_net.network.train(sample[0], q_target)
      
   Q_net.update_delay_target_network()

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










