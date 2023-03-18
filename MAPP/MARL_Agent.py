import tensorflow as tf
import numpy as np
from MARL_DQN import MARL_DQN
import ModifiedTensorBoard
import time

class MARL_Agent(tf.keras.layers.Layer):
    def __init__(self, num_actions, model_name):
        super().__init__()

        self.network = MARL_DQN(num_actions)
        self.delay_target_network = MARL_DQN(num_actions)
        self.update_delay_target_network()

        self.num_actions = num_actions

        self.tensorboard = ModifiedTensorBoard.ModifiedTensorBoard(log_dir="logs/{}-{}".format(model_name, int(time.time())))

    def __call__(self, x, training = False):
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
        #FIXME: without it gives me an dimension error with this it gives a invalid value for attribute error
        observation = tf.expand_dims(observation,0)
        q_values = self(observation)
        action = [np.argmax(q_values[i]) if np.random.rand() > epsilon else np.random.randint(self.num_actions) for i in range(self.num_environments)]
        return action


    def update_delay_target_network(self):
        """
        Sets the weights of Delay-Target-Network.
        """
        self.delay_target_network.set_weights(self.network.get_weights())

    #FIXME: the docstring is wrong (from an old version)
    def q_target_array(self, ERP, sample_number, discount_factor = 0.95):
        """
        Calculates Q-target (expected reward) with Delay-Target-Network.

        Parameters:
            sample (list): list of lists each filled with [observation_t, action, reward, observation]
            discount_factor (float): to set the influence of future rewards

        Returns:
            q_target (float): expected reward from "optimal" action
        """
        observation = ERP.observation[sample_number]
        reward = ERP.reward[sample_number]

        q_values = self.delay_target_network(observation)
        max_q_value = tf.math.top_k(q_values, k=1, sorted=True)
        q_target = reward + discount_factor * max_q_value.values.numpy()
        return q_target


    def training(self, data):
        for batch in data:
            observation, action, reward, next_observation = batch
            q_target = self.q_target(reward, next_observation)
            self.network.train(observation, action, q_target)