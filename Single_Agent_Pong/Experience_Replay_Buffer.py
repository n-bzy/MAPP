import numpy as np
import tensorflow as tf

class ExperienceReplayBuffer():
    def __init__(self, size): 

        self.size = size 
        self.experience_replay_buffer = None 
        self.observation = np.ndarray(shape=(self.size, 4, 84, 84, 1), dtype=np.float32)
        self.action = np.ndarray(shape = (self.size, ), dtype=np.int64)
        self.reward = np.ndarray(shape = (self.size, ))
        self.next_observation = np.ndarray(shape=(self.size, 4, 84, 84, 1), dtype=np.float32)
        self.index = 0

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
        data = data.cache().batch(128).shuffle(500).prefetch(tf.data.AUTOTUNE) #wann batchen wir??? erst und dann shuffle oder so wie es jetzt ist????

        return data