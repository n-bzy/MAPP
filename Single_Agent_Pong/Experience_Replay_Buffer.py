import numpy as np
import tensorflow as tf

class ExperienceReplayBuffer():
    def __init__(self, size): 

        self.size = size 
        self.experience_replay_buffer = [0] * self.size
        self.experience_replay_buffer_samples = []
        self.index = 0

    def sample(self, batch_size = 32):
        """
        Randomly choose a sample from the Experience-Replay-Buffer.

        Parameters: 
            batch_size (int): amount of samples to be trained on later on
        """
        for _ in range(batch_size):
            index = np.random.randint(self.size)
            sample = self.experience_replay_buffer[index] 
            self.experience_replay_buffer_samples.append(sample)


    def fill(self, environment):
        """
        Fill up the Experience Replay Buffer with random samples (observation_t, action, reward, observation) from the environment.

        Parameters: 
            environment (gymnasium): the environment to get observations, take actions and get reward
        """

        observation, _ = environment.reset()
        observation = self.normalizing(observation)
        for _ in range(self.size):
            action =  np.random.randint(6)

            next_observation, reward, terminated, truncated, _ = environment.step(action)
            next_observation = self.normalizing(next_observation)

            self.experience_replay_buffer[self.index] = (observation, action, tf.cast(reward, tf.float32), next_observation)

            if truncated == True or terminated == True: 
                next_observation = environment.reset()[0]
                next_observation = self.normalizing(next_observation)

            
            observation = next_observation

            self.set_index()
    
    def preprocessing(self):
        """
        Converts list consisting of 32 experiences to tf.data.Dataset and preprocesses it.
        Takes about 0.04 s on average

        Returns:
            data (tf.data.Dataset): dataset to train on
        """
        data = tf.data.experimental.from_list(self.experience_replay_buffer_samples)
        data = data.cache().shuffle(500).batch(32).prefetch(tf.data.AUTOTUNE)

        self.experience_replay_buffer_samples = []

        return data
    
    def set_index(self):
        """
        Sets the index of the ERP, to enable the overwriting of old experiences.
        """
        self.index += 1
        if self.index == self.size:
            self.index = 0


    def normalizing(self, observation):
        """
        Normalizes an observation and casts it to a tf.float.
        """
        observation = tf.cast(observation, tf.float32)  / 255.
        return observation
