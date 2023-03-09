import gymnasium as gym
from Experience_Replay_Buffer import ExperienceReplayBuffer
from Agent import Agent


num_environments = 32
num_actions = 6 #env.action_space
m = 50 # amount of training samples
timesteps = 30 # amount of samples to fill ERP with after the its filled up once
ERP_size = 100

# instantiate environments
env_name = 'ALE/Pong-v5' 
environments = [gym.make(env_name) for _ in range(num_environments)] #render_mode = 'human')

# instantiate q_network
Q_net = Agent(num_actions, num_environments)
Q_net.update_delay_target_network()

ERP =  ExperienceReplayBuffer(size = 100) 

reward_per_episode = []

for episode in range(10):

   reward_of_episode = Q_net.fill(environments, timesteps, ERP)

   for amount_of_samples in range(m):
      sample = ERP.sample()
      q_target = Q_net.q_target(sample) 
      observations = [sample[0] for sample in sample]
      Q_net.network.train(observations, q_target)
      
   Q_net.update_delay_target_network()

   reward_per_episode.append(reward_of_episode)

   print(f'done with epsiode {episode} with reward {reward_of_episode}') 