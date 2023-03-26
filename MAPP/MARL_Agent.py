import tensorflow as tf
import numpy as np
from MARL_DQN import MARL_DQN
import ModifiedTensorBoard
import time

class MARL_Agent(tf.keras.Model):
    def __init__(self, environment, ERP, num_actions, model_name, epsilon = 1, min_epsilon = 0.001, epsilon_decay_value = 0.999985):
        super().__init__()

        self.network = MARL_DQN(num_actions)
        self.delay_target_network = MARL_DQN(num_actions)
        self.update_delay_target_network()

        self.environment = environment
        self.ERP = ERP

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_value = epsilon_decay_value

        self.num_actions = num_actions
        self.metrics_list = [tf.keras.metrics.Mean(name="loss")]

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

    def epsilon_greedy_sampling(self, observation):
        """
        Epsilon-greedy sampling to balance exploration and exploitation.

        Parameters:
            observation (ndarray): observation
            epsilon (float): probability for exploration

        Returns:
            action (int): either action with highest Q-value (exploitation) or random action (exploration)
        """
        if np.random.rand() > self.epsilon:
            q_values = self(tf.expand_dims(observation, 0))
            action = np.argmax(q_values).item()
        else:
            action = np.random.randint(self.num_actions)

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
        max_q_value = tf.math.reduce_max(q_values, axis = 1)
        q_target = reward + discount_factor * max_q_value
        return q_target


    def training(self):
        """
        Train the Agent on data sampled from the ERP.
        """
        self.ERP.sample()
        data = self.ERP.preprocessing()

        for batch in data:
            observation, action, reward, next_observation = batch
            q_target = self.q_target(reward, next_observation)
            self.network.train(observation, action, q_target)


    def epsilon_decay(self):
        """
        Decay epsilon.

        Parameters:
            MIN_EPSILON (float): a threshold which epsilon may not fall below
            EPSILON_DECAY (float): the factor epsilon is decayed with
        """
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay_value
            self.epsilon = max(self.min_epsilon, self.epsilon)