import numpy as np
import tensorflow as tf

class ExperienceReplayBuffer():
    def __init__(self, size): 

        self.size = size 
        self.experience_replay_buffer = [0] * self.size
        self.experience_replay_buffer_new_samples = []
        self.observation = np.ndarray(shape=(self.size, 4, 84, 84, 1), dtype=np.float32)
        self.action = np.ndarray(shape = (self.size, ), dtype=np.int64)
        self.reward = np.ndarray(shape = (self.size, ))
        self.next_observation = np.ndarray(shape=(self.size, 4, 84, 84, 1), dtype=np.float32)
        self.index = 0

    def sample(self, batch_size = 32):
        """
        Randomly choose a sample from the Experience-Replay-Buffer.
        """
        for _ in range(batch_size):
            index = np.random.randint(self.size)
            print(index)
            sample = self.experience_replay_buffer[index] 
            self.experience_replay_buffer_new_samples.append(sample)

    def fill_up(self, environment):
        """
        Fill up the whole ERP with samples from random actions.
        """
        self.observation[self.index] = environment.reset()[0]
        for i in range(self.size):
            if i > 0:
                self.observation[self.index] = self.next_observation[self.index - 1]

            self.action[self.index] = np.random.randint(6)

            self.next_observation[self.index], self.reward[self.index], terminated, truncated, info = environment.step(self.action[self.index])

            self.index += 1
        self.index = 0

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

        observation = environment.reset()[0]
        for _ in range(self.size):
            action =  np.random.randint(6)

            next_observation, reward, terminated, truncated, info = environment.step(action)

            self.experience_replay_buffer[self.index] = (observation, action, reward, next_observation)

            if truncated == True or terminated == True: 
                next_observation = environment.reset()[0]
            
            observation = next_observation

            self.index += 1
        self.index = 0

    def preprocessing(self):
        """
        Transforms arrays to tf.data.Dataset
        """

        observations = tf.data.Dataset.from_tensor_slices(self.observation)
        actions = tf.data.Dataset.from_tensor_slices(self.action)
        rewards = tf.data.Dataset.from_tensor_slices(self.reward)
        next_observations = tf.data.Dataset.from_tensor_slices(self.next_observation)

        data = tf.data.Dataset.zip((observations, actions, rewards, next_observations))
        data = data.map(lambda x,y,z,t: (tf.cast(x, tf.float32)  / 256., y, tf.cast(z, tf.float32), tf.cast(t, tf.float32) / 256.))
        data = data.cache().shuffle(500).batch(32).prefetch(tf.data.AUTOTUNE) #wann batchen wir??? erst und dann shuffle oder so wie es jetzt ist????

        return data
    
    def preprocessing_list(self):
        # https://www.tensorflow.org/api_docs/python/tf/data/experimental/from_list 
        # we can work with lists and create the dataset like this

        #tf.print(self.experience_replay_buffer_new_samples)

        print(type(self.experience_replay_buffer_new_samples))
        data = tf.data.experimental.from_list(self.experience_replay_buffer_new_samples)
        data = data.map(lambda x,y,z,t: (tf.cast(x, tf.float32)  / 255., y, tf.cast(z, tf.float32), tf.cast(t, tf.float32) / 255.))
        data = data.cache().shuffle(500).batch(32).prefetch(tf.data.AUTOTUNE) #wann batchen wir??? erst und dann shuffle oder so wie es jetzt ist????

    
        self.experience_replay_buffer_new_samples = []

        return data
    
    def set_index(self):
        self.index += 1
        if self.index == self.size:
            self.index = 0
