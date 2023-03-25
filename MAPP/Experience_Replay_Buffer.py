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


    def fill(self, env):
        """
        Fill up the Experience Replay Buffer with random samples (observation_t, action, reward, observation) from the environment.

        Parameters:
            env (pettingzoo): the environment to get observations, take actions and get reward
        """

        for _ in range(self.size):

            env.reset()

            # agent gets removed from env.agents when it is flagged as done
            while env.agents:
                for agent in env.agent_iter():

                    observation, reward, termination, truncation, info = env.last()
                    observation = self.normalizing(observation)
                    action = np.random.randint(6)
                    env.step(action)

                    #FIXME: last() only returns the current observation, and step() returns nothing.
                    # calling last() again after doing a step seems to be the only way of getting the next observation
                    next_observation, _, _, _, _ = env.last()
                    next_observation = self.normalizing(next_observation)

                    self.experience_replay_buffer[self.index] = (observation, action, tf.cast(reward, tf.float32), next_observation)
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

    







