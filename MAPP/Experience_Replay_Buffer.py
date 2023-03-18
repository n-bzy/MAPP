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
        self.experience_replay_buffer_new_samples = []

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

            self.experience_replay_buffer_new_samples.append((observation, action, reward, next_observation))
            observation = next_observation

        end = time.time()
        #print(f"duration fill up ERP of size {self.size} list version: ", end-start)
        return end-start


    '''def preprocessing(self, observation):
        observation = tf.cast(observation, tf.float32) / 256. #np.shape(4, 84, 84, 1)
        #observation = tf.expand_dims(observation, 0) #np.shape(1, 4, 84, 84, 1)
        return observation'''
    
    def preprocessing(self):
        # https://www.tensorflow.org/api_docs/python/tf/data/experimental/from_list 
        # we can work with lists and create the dataset like this


        observations = tf.data.Dataset.from_tensor_slices(self.observation)
        actions = tf.data.Dataset.from_tensor_slices(self.action)
        rewards = tf.data.Dataset.from_tensor_slices(self.reward)
        next_observations = tf.data.Dataset.from_tensor_slices(self.next_observation)

        data = tf.data.Dataset.zip((observations, actions, rewards, next_observations))
        data = data.map(lambda x,y,z,t: (tf.cast(x, tf.float32)  / 256., y, tf.cast(z, tf.float32), tf.cast(t, tf.float32) / 256.))
        data = data.cache().batch(128).shuffle(500).prefetch(tf.data.AUTOTUNE) #wann batchen wir??? erst und dann shuffle oder so wie es jetzt ist????

        return data
    


    def preprocessing_list(self):
        # https://www.tensorflow.org/api_docs/python/tf/data/experimental/from_list 
        # we can work with lists and create the dataset like this

        data = tf.data.experimental.from_list(self.experience_replay_buffer_new_samples)
        data = data.map(lambda x,y,z,t: (tf.cast(x, tf.float32)  / 256., y, tf.cast(z, tf.float32), tf.cast(t, tf.float32) / 256.))
        data = data.cache().batch(128).shuffle(500).prefetch(tf.data.AUTOTUNE) #wann batchen wir??? erst und dann shuffle oder so wie es jetzt ist????

        self.experience_replay_buffer_new_samples = []

        return data
    
    '''
    First shuffling, then batching
    done with epsiode 100 with reward -9.0
    done with epsiode 101 with reward -9.0
    done with epsiode 102 with reward -11.0
    done with epsiode 103 with reward -11.0
    done with epsiode 104 with reward -7.0
    done with epsiode 105 with reward -10.0
    done with epsiode 106 with reward -13.0

    first batching, then shuffling
    done with epsiode 100 with reward -5.0
    done with epsiode 101 with reward -10.0
    done with epsiode 102 with reward -5.0
    done with epsiode 103 with reward -11.0
    done with epsiode 104 with reward -11.0
    done with epsiode 105 with reward -11.0
    done with epsiode 106 with reward -11.0

    '''

    







