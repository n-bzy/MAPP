import tensorflow as tf
import numpy as np
from DQN import DQN
import ModifiedTensorBoard
import time

class Agent(tf.keras.layers.Layer):
    def __init__(self, environment, ERP, model_name, num_actions = 6, epsilon = 1):
        super().__init__()

        self.network = DQN(num_actions) 
        self.delay_target_network = DQN(num_actions) 

        self.metrics_list = [tf.keras.metrics.Mean(name="loss")] 

        self.ERP = ERP
        self.environment = environment

        self.epsilon = epsilon
        self.reward_of_episode = 0
        self.num_actions = num_actions

        self.tensorboard = ModifiedTensorBoard.ModifiedTensorBoard(log_dir="logs/{}-{}".format(model_name, int(time.time())))

    def call(self, x, training = False):
        """
        Calls Deep Q-Network to get Q-values. 

        Parameters: 
            x (ndarray): observation
        Returns: 
            x (ndarray): Q-value for every action 
        """
        x = self.network(x, training) 
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
            action = np.argmax(q_values).item()
        else:
            action = np.random.randint(self.num_actions)

        return action
    
    def play(self, observation):
        """
        Select an action to take one step in the environment and store the experience in ERP. 
        Decay epsilon.

        Parameters:
            observation (np.ndarray): the current observation of the environment

        Returns:
            next_observation (np.ndarray): the current observation of the environment
            terminated (boole): tells whether one player won the game
            truncated (boole): tells whether ??????????????????????? frames were displayed
        """
        action = self.epsilon_greedy_sampling(observation, epsilon = self.epsilon)
        next_observation, reward, terminated, truncated, _ = self.environment.step(action)

        self.ERP.experience_replay_buffer[self.ERP.index] = (observation, action, reward, next_observation)
        self.ERP.set_index()

        self.epsilon_decay()

        self.reward_of_episode += reward

        return next_observation, terminated, truncated
    
    def q_target(self, reward, next_observation, discount_factor = 0.99):
        """
        Calculates Q-target (expected reward) with Delay-Target-Network.

        Parameters: 
            sample (list): list of lists each filled with [observation_t, action, reward, observation] 
            discount_factor (float): to set the influence of future rewards 

        Returns: 
            q_target (float): expected reward from "optimal" action 
        """
        q_values = self.delay_target_network(next_observation)
        max_q_value = tf.math.reduce_max(q_values, axis = 1) #returns maximum for each batch
        q_target = reward + discount_factor * max_q_value
        return q_target
    
    def training(self):
        """
        Train the Agent on data sampled from the ERP.
        """
        self.ERP.sample()
        data = self.ERP.preprocessing_list()

        for batch in data:
            observation, action, reward, next_observation = batch
            q_target = self.q_target(reward, next_observation)
            self.network.train(observation, action, q_target)


    def update_delay_target_network(self):
        """
        Sets the weights of Delay-Target-Network.
        """
        self.delay_target_network.set_weights(self.network.get_weights())

    def epsilon_decay(self, MIN_EPSILON = 0.001, EPSILON_DECAY = 0.999985):
        """
        Decay epsilon.

        Parameters:
            MIN_EPSILON (float): a threshold which epsilon may not fall below 
            EPSILON_DECAY (float): the factor epsilon is decayed with
        """
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)