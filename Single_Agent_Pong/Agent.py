import tensorflow as tf
import numpy as np
from DQN import DQN
import ModifiedTensorBoard
import time

class Agent(tf.keras.layers.Layer):
    def __init__(self, num_actions, model_name, epsilon = 1):
        super().__init__()

        self.network = DQN(num_actions) 
        self.delay_target_network = DQN(num_actions) 

        self.metrics_list = [tf.keras.metrics.Mean(name="loss")]

        self.num_actions = num_actions

        self.epsilon = epsilon

        self.reward_of_episode = 0

        self.tensorboard = ModifiedTensorBoard.ModifiedTensorBoard(log_dir="logs/{}-{}".format(model_name, int(time.time())))

    def call(self, x, training = False):
        """
        Calls Deep Q-Network to get Q-values. 

        Parameters: 
            x (ndarray): observation
        Returns: 
            x (ndarray): Q-value for every action 
        """
        x = self.network(x) 
        return x
    
    def epsilon_greedy_sampling(self, observation, epsilon = 0.05):
        """
        Epsilon-greedy sampling to balance exploration and exploitation. 

        Parameters: 
            observation (ndarray): observation
            epsilon (float): probability for exploration

        Returns:
            action (int): either action with highest Q-value (exploitation) or random action (exploration)
        """
        if np.random.rand() > epsilon:
            q_values = self(tf.expand_dims(tf.cast(observation, tf.float32) / 255., 0))
            action = np.argmax(q_values)
        else:
            action = np.random.randint(self.num_actions)
        return action
    
    def old_play(self, environment, timesteps, ERP, epsilon):
        """
        Fill up the Experience Replay Buffer with samples from the environment.

        Parameters: 
            environment (gymnasium): the environment to get observations, take actions and get reward
            timesteps (int): amount of new samples 
            ERP (Experience_Replay_Buffer): store the samples of size num_environments filled with 
                            [observation_t, action, reward, observation]
            epsilon (float): ???????????????

        Returns: 
            reward_of_epsiode (int): reward of this episode
        """
        reward_of_episode = 0

        ERP.observation[ERP.index] = environment.reset()[0] 
        
        for i in range(timesteps):
            ERP.action[ERP.index] =  self.epsilon_greedy_sampling(observation = ERP.observation[ERP.index], epsilon = epsilon)

            ERP.next_observation[ERP.index], ERP.reward[ERP.index], terminated, truncated, info = environment.step(ERP.action[ERP.index])

            reward_of_episode += ERP.reward[ERP.index]

            if truncated == True or terminated == True:
                break

            ERP.set_index()

            ERP.observation[ERP.index] = ERP.next_observation[ERP.index - 1]
            
        return reward_of_episode
    
    def play(self, observation, environment, ERP):
        
        action =  self.epsilon_greedy_sampling(observation, epsilon = self.epsilon)

        next_observation, reward, terminated, truncated, info = environment.step(action)

        ERP.experience_replay_buffer[ERP.index] = (observation, action, reward, next_observation)

        ERP.set_index()

        self.epsilon_decay()

        self.reward_of_episode += reward

        return next_observation, terminated, truncated
    
    def q_target(self, reward, next_observation, discount_factor = 0.95):
        """
        Calculates Q-target (expected reward) with Delay-Target-Network.

        Parameters: 
            sample (list): list of lists each filled with [observation_t, action, reward, observation] 
            discount_factor (float): to set the influence of future rewards 

        Returns: 
            q_target (float): expected reward from "optimal" action 
        """
        q_values = self.delay_target_network(next_observation) #no time_distributed leaves shape = (4,6), mit time_distirbuted shape = (1,6)
        max_q_value = tf.math.reduce_max(q_values, axis = 1) #returns maximum for each batch
        q_target = reward + discount_factor * max_q_value #* 265.
        return q_target
    
    def training(self, data):
        for batch in data:
            observation, action, reward, next_observation = batch
            q_target = self.q_target(reward, next_observation)
            self.network.train(observation, action, q_target)
            print("trained")


    def update_delay_target_network(self):
        """
        Sets the weights of Delay-Target-Network.
        """
        self.delay_target_network.set_weights(self.network.get_weights())

    def epsilon_decay(self, MIN_EPSILON = 0.001, EPSILON_DECAY = 0.999985):
        # decay epsilon
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)