import tensorflow as tf
import numpy as np
from DQN import DQN

class Agent(tf.keras.layers.Layer):
    def __init__(self, num_actions):
        super().__init__()

        self.network = DQN(num_actions) 
        self.delay_target_network = DQN(num_actions) 

        self.metrics_list = [tf.keras.metrics.Mean(name="loss")]

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
        probability = np.random.rand() 
        if probability > epsilon:
            action = np.argmax(self(observation))
        else: 
            action = np.random.randint(6)
        return action
    

    def q_target(self, sample, discount_factor = 0.95):
        """
        Calculates Q-target (expected reward) with Delay-Target-Network.

        Parameters: 
            sample (list): [observation_t, action, reward, observation] 
            discount_factor (float): to set the influence of future rewards 

        Returns: 
            q_target (float): expected reward from "optimal" action 
        """
        q_values = self.delay_target_network(sample[3])
        max_q_value = tf.math.top_k(q_values, k=1, sorted=True) 
        q_target = sample[2] + discount_factor * max_q_value.values.numpy() 
        return q_target
    
    def update_delay_target_network(self):
        """
        Sets the weights of Delay-Target-Network.
        """
        self.delay_target_network.set_weights(self.network.get_weights())
    


 