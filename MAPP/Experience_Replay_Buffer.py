import numpy as np
import tensorflow as tf

class ExperienceReplayBuffer():
    def __init__(self, size): 

        self.size = size 
        #self.experience_replay_buffer = [0] * self.size # maybe an array is nice
        self.observation = np.ndarray(shape=(self.size, 4, 84, 84, 1), dtype=np.float32)
        self.action = np.ndarray(shape = (self.size, ), dtype=np.int64)
        self.reward = np.ndarray(shape = (self.size, ))
        self.next_observation = np.ndarray(shape=(self.size, 4, 84, 84, 1), dtype=np.float32)
        self.index = 0

    def sample(self):
        """
        Randomly choose a sample from the Experience-Replay-Buffer.
        """
        sample_number = np.random.randint(self.size) 
        return sample_number
    
    '''
    def sample(self):
        """
        Randomly choose a sample from the Experience-Replay-Buffer.
        """
        sample = self.experience_replay_buffer[np.random.randint(self.size)] 
        return sample
    '''
    

    def fill_up(self, environment):
        """
        Fill up the whole ERP with random actions
        """
        self.observation[self.index] = environment.reset()[0] # do we need this every time?????????????????????????????????????
        for i in range(self.size):
            if i > 0:
                self.observation[self.index] = self.next_observation[self.index - 1]

            self.action[self.index] =  np.random.randint(6)

            self.next_observation[self.index], self.reward[self.index], terminated, truncated, info = environment.step(self.action[self.index])

            self.index += 1
        self.index = 0

    def preprocessing(self, observation):
        observation = tf.cast(observation, tf.float32) / 256. #np.shape(4, 84, 84, 1)
        #observation = tf.expand_dims(observation, 0) #np.shape(1, 4, 84, 84, 1)
        return observation

    







