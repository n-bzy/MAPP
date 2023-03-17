import numpy as np
import tensorflow as tf
import time

class ExperienceReplayBuffer():
    def __init__(self, size): 

        self.size = size 
        #self.experience_replay_buffer = [0] * self.size # maybe an array is nice
        self.experience_replay_buffer = None
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
        start = time.time()
        self.observation[self.index] = environment.reset()[0] # do we need this every time?????????????????????????????????????
        for i in range(self.size):
            if i > 0:
                self.observation[self.index] = self.next_observation[self.index - 1]

            self.action[self.index] =  np.random.randint(6)

            self.next_observation[self.index], self.reward[self.index], terminated, truncated, info = environment.step(self.action[self.index])

            self.index += 1
        self.index = 0
        end = time.time()
        #print(f"duration fill up ERP of size {self.size} array version: ", end-start)
        return end-start



    def fill(self, environment):
        """
        Fill up the Experience Replay Buffer with samples from the environment. The whole one in the first episode, afterward replace #timesteps samples.

        Parameters: 
            environments (gymnasium): the environment to get observations, take actions and get reward
            timesteps (int): amount of new samples 
            ERP (Experience_Replay_Buffer): store the samples of size num_environments filled with 
                            [observation_t, action, reward, observation]

        Returns: 
            reward_of_epsiode (list): reward of this episode in each environment 
        """
        start = time.time()
        observation = environment.reset()[0]
        for _ in range(self.size):
            action =  np.random.randint(6)

            next_observation, reward, terminated, truncated, info = environment.step(action)

            self.experience_replay_buffer[self.index] = [observation, action, reward, next_observation]
            observation = next_observation
            #@ToDo: Denkfehler: truncated und terminated nur im letzen Schritt

            self.index += 1
        self.index = 0
        end = time.time()
        #print(f"duration fill up ERP of size {self.size} list version: ", end-start)
        return end-start


    '''def preprocessing(self, observation):
        observation = tf.cast(observation, tf.float32) / 256. #np.shape(4, 84, 84, 1)
        #observation = tf.expand_dims(observation, 0) #np.shape(1, 4, 84, 84, 1)
        return observation'''
    
    def preprocessing(self):
        observations = tf.data.Dataset.from_tensor_slices(self.observation)
        actions = tf.data.Dataset.from_tensor_slices(self.action)
        rewards = tf.data.Dataset.from_tensor_slices(self.reward)
        next_observations = tf.data.Dataset.from_tensor_slices(self.next_observation)

        data = tf.data.Dataset.zip((observations, actions, rewards, next_observations))
        data = data.map(lambda x,y,z,t: (tf.cast(x, tf.float32)  / 256., y, tf.cast(z, tf.float32), tf.cast(t, tf.float32) / 256.))
        data = data.cache().shuffle(500).batch(32).prefetch(tf.data.AUTOTUNE) #wann batchen wir??? erst und dann shuffle oder so wie es jetzt ist????
        
        return data

    







