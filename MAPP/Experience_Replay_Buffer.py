import numpy as np

class ExperienceReplayBuffer():
    def __init__(self, size): 

        self.size = size 
        self.experience_replay_buffer = [0] * self.size # maybe an array is nice
        self.index = 0

    def sample(self):
        """
        Randomly choose a sample from the Experience-Replay-Buffer.
        """
        sample = self.experience_replay_buffer[np.random.randint(self.size)] 
        return sample
    
    def fill(self, environment, Q_net, timesteps):
        """
        Fill up the Experience Replay Buffer. The whole one in the first episode, afterward replace #timesteps samples.

        Parameters: 
            environment (gymnasium): the environment to get observations, take actions and get reward 
            Q_net (Agent): decides which action to take 
            timesteps (int): amount of new samples 

        Returns: 
            experience_replay_buffer (list): list of lists filled with [observation_t, action, reward, observation]
        """
        if self.experience_replay_buffer == [0] * self.size:
            timesteps = self.size
        observation, _ = environment.reset() # state = observation, do we need this every time?????????????????????????????????????

        reward_of_episode = 0
        for timestep in range(timesteps):
            observation_t = observation # observation_t represents state_t 
            action =  Q_net.epsilon_greedy_sampling(observation = observation_t, epsilon = 0.05)
            observation, reward, terminated, truncated, info = environment.step(action)
            reward_of_episode += reward
            self.experience_replay_buffer[self.index] = [observation_t, action, reward, observation]
            self.index += 1
            if self.index == self.size:
                self.index = 0
            if truncated == True or terminated == True and timesteps != self.size:
                break
        return reward_of_episode

    







