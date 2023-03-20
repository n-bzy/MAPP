import numpy as np
import tensorflow as tf

class ExperienceReplayBuffer():
    def __init__(self, size): 

        self.size = size 
        self.experience_replay_buffer = [0] * self.size
        self.experience_replay_buffer_new_samples = []
        self.index = 0

    def sample(self, batch_size = 32):
        """
        Randomly choose a sample from the Experience-Replay-Buffer.

        Troubles: 
            duplicates no problem
                visited = set()
                a = {x for x in indexes if x in visited or (visited.add(x) or False)}
                print(a)
            last observation before finish no problem

            last experience no problem
        """
        #indexes = []
        for _ in range(batch_size):
            index = np.random.randint(self.size)
            #indexes.append(index)
            sample = self.experience_replay_buffer[index] 
            self.experience_replay_buffer_new_samples.append(sample)
        #print(indexes)


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

            self.set_index()
    
    def preprocessing_list(self):
        # https://www.tensorflow.org/api_docs/python/tf/data/experimental/from_list 
        # we can work with lists and create the dataset like this

        #tf.print(self.experience_replay_buffer_new_samples)

        #print(type(self.experience_replay_buffer_new_samples))
        #print(self.index)
        data = tf.data.experimental.from_list(self.experience_replay_buffer_new_samples)
        data = data.map(lambda x,y,z,t: (tf.cast(x, tf.float32)  / 255., y, tf.cast(z, tf.float32), tf.cast(t, tf.float32) / 255.))
        data = data.cache().shuffle(500).batch(32).prefetch(tf.data.AUTOTUNE) #wann batchen wir??? erst und dann shuffle oder so wie es jetzt ist????

    
        self.experience_replay_buffer_new_samples = []

        return data
    
    def set_index(self):
        self.index += 1
        if self.index == self.size:
            self.index = 0
