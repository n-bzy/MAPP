import tensorflow as tf
import numpy as np
from DQN import DQN
import ModifiedTensorBoard
import time

class Agent(tf.keras.layers.Layer):
    def __init__(self, num_actions, num_environments, model_name):
        super().__init__()

        self.network = DQN(num_actions) 
        self.delay_target_network = DQN(num_actions) 

        self.metrics_list = [tf.keras.metrics.Mean(name="loss")]

        self.num_environments = num_environments
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
        q_values = self(observation)
        action = [np.argmax(q_values[i]) if np.random.rand() > epsilon else np.random.randint(self.num_actions) for i in range(self.num_environments)]
        return action
    


    
    def fill(self, environments, timesteps, ERP, epsilon):
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
        if ERP.experience_replay_buffer == [0] * ERP.size:
            timesteps = ERP.size
        observation = [environment.reset()[0] for environment in environments]# state = observation, do we need this every time?????????????????????????????????????

        print(type(observation))
        reward_of_episode = [0] * len(environments)
        for _ in range(timesteps):
            action =  self.epsilon_greedy_sampling(observation = observation, epsilon = epsilon)

            batch = []
            for i in range(len(environments)):
                next_observation, reward, terminated, truncated, info = environments[i].step(action[i])
                reward_of_episode[i] += reward
                batch.append([observation[i], action, reward, next_observation])
                observation[i] = next_observation
                #@ToDo: Denkfehler: truncated und terminated nur im letzen Schritt
                if truncated == True or terminated == True and timesteps != ERP.size:
                    break
            ERP.experience_replay_buffer[ERP.index] = batch
            ERP.index += 1
            if ERP.index == ERP.size:
                ERP.index = 0
            

        return reward_of_episode
    
    def fill_array(self, environment, timesteps, ERP, epsilon):
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

        ERP.observation[ERP.index] = environment.reset()[0] # do we need this every time?????????????????????????????????????
        ERP.observation[ERP.index] = ERP.preprocessing(ERP.observation[ERP.index])
        for i in range(timesteps):
            if i > 0:
                ERP.observation[ERP.index] = ERP.next_observation[ERP.index - 1]
            #print(type(self.epsilon_greedy_sampling(observation = ERP.observation[i + ERP.index], epsilon = epsilon)[0])) # np.int64
            ERP.action[ERP.index] =  self.epsilon_greedy_sampling(observation = ERP.observation[ERP.index], epsilon = epsilon)[0]
            #print(type(ERP.action[ERP.index]))

            ERP.next_observation[ERP.index], ERP.reward[ERP.index], terminated, truncated, info = environment.step(ERP.action[ERP.index])
            ERP.next_observation[ERP.index] = ERP.preprocessing(ERP.next_observation[ERP.index])
            reward_of_episode += ERP.reward[ERP.index]

            if truncated == True or terminated == True:
                break

            ERP.index += 1
            if ERP.index == ERP.size:
                ERP.index = 0
            
        return reward_of_episode
    

    def fill_array_multiple_environments(self, environments, timesteps, ERPs, epsilon):
        """
        Fill up the Experience Replay Buffer of each environment with samples from the environment.

        Parameters: 
            num_environments (int): amoutn of environments
            environments (list): environments contains num_environments environments (gymnasium) to get observations, take actions and get reward
            timesteps (int): amount of new samples 
            ERPs (list): ERPs contains an experience replay buffer (Experience_Replay_Buffer)for each environment
            epsilon (float): ???????????????

        Returns: 
            reward_of_epsiode (list): reward of this episode in each environment 
        """
        reward_of_episode = np.zeros(shape=(self.num_environments))

        for i in range(self.num_environments):
            ERPs[i].observation[ERPs[i].index] = ERPs[i].preprocessing(environments[i].reset()[0])

        for i in range(timesteps):
            if i > 0:
                for i in range(self.num_environments):
                    ERPs[i].observation[ERPs[i].index] = ERPs[i].next_observation[ERPs[i].index - 1]
            observations = tf.stack([ERPs[i].observation[ERPs[i].index] for i in range(self.num_environments)]) #shape = (num_environments, 4, 84, 84, 1)
            
            actions =  self.epsilon_greedy_sampling(observation = observations, epsilon = epsilon)

            for i in range(self.num_environments):
                ERPs[i].action[ERPs[i].index] = actions[i]
                ERPs[i].next_observation[ERPs[i].index], ERPs[i].reward[ERPs[i].index], terminated, truncated, info = environments[i].step(ERPs[i].action[ERPs[i].index])
                ERPs[i].next_observation[ERPs[i].index] = ERPs[i].preprocessing(ERPs[i].next_observation[ERPs[i].index])
                reward_of_episode[i] += ERPs[i].reward[ERPs[i].index]

                if truncated == True or terminated == True:
                    break

                ERPs[i].index += 1
                if ERPs[i].index == ERPs[i].size:
                    ERPs[i].index = 0

        print(f"{timesteps} samples replaced in every ERP")
            
        return reward_of_episode


    # started writing a specific fill method for the multi agent case
    # this doesn't include a section for filling the ERP at the beginning.
    # I thought we can do that in a seperate function maybe.
    def multi_agent_fill(self, env, ERP, epsilon):

        observations = env.reset()
        reward_of_episode = [0] * len(2) # currently hard coded for the amount of agents we have

        # Initialize the episode rewards and done flags for each agent
        dones = {agent: False for agent in env.agents}
        i = 0

        while not all(dones.values()):
            action = self.epsilon_greedy_sampling(observation = observations, epsilon = epsilon)

            batch = []
            next_observation, reward, dones, info = env.step(action)
            reward_of_episode[i] += reward
            batch.append([observations[i], action, reward, next_observation])
            observations[i] = next_observation

            ERP.experience_replay_buffer[ERP.index] = batch
            ERP.index += 1
            if ERP.index == ERP.size:
                ERP.index = 0
            i = (i +1) % 2 # hard coded for two agents again
        return reward_of_episode

    # currently just for the multi-agent case as a replacement for the way it is handled in the single-agent fill method
    # number_of_entries should be smaller than the max size of the ERP!
    # this doesn't really look better than putting everything in the same function though...
    # too much duplication of code to feel cleaner, tbh.
    def fill_ERP(self, env, ERP, epsilon, number_of_entries):

        filled = False
        while not filled:
            observations = env.reset()
            dones = {agent: False for agent in env.agents}
            i = 0

            while not all(dones.values()):
                action = self.epsilon_greedy_sampling(observation = observations, epsilon = epsilon)

                batch = []
                next_observation, reward, dones, info = env.step(action)
                batch.append([observations[i], action, reward, next_observation])
                observations[i] = next_observation

                ERP.experience_replay_buffer[ERP.index] = batch
                ERP.index += 1
                if ERP.index == ERP.size:
                    ERP.index = 0
                i = (i +1) % 2 # hard coded for two agents again

            if ERP.size >= number_of_entries:
                filled = True





    
    
    def q_target(self, sample, discount_factor = 0.95):
        """
        Calculates Q-target (expected reward) with Delay-Target-Network.

        Parameters: 
            sample (list): list of lists each filled with [observation_t, action, reward, observation] 
            discount_factor (float): to set the influence of future rewards 

        Returns: 
            q_target (float): expected reward from "optimal" action 
        """
        observation = [sample[3] for sample in sample]
        reward = [sample[2] for sample in sample]

        q_values = self.delay_target_network(observation)
        max_q_value = [tf.math.top_k(q_values[i], k=1, sorted=True) for i in range(len(sample))]
        q_target = [reward[i] + discount_factor * max_q_value[i].values.numpy() for i in range(len(sample))]
        return q_target
    
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
    



    def q_target_multiple_environments(self, ERPs, sample_numbers, discount_factor = 0.95):
        """
        Calculates Q-target (expected reward) with Delay-Target-Network.

        Parameters: 
            sample (list): list of lists each filled with [observation_t, action, reward, observation] 
            discount_factor (float): to set the influence of future rewards 

        Returns: 
            q_target (float): expected reward from "optimal" action 
        """
        observation = tf.stack([ERPs[i].observation[sample_numbers[i]] for i in range(self.num_environments)])
        reward = [ERPs[i].reward[sample_numbers[i]] for i in range(self.num_environments)]

        q_values = self.delay_target_network(observation)
        max_q_value = [tf.math.top_k(q_values[i], k=1, sorted=True) for i in range(self.num_environments)]
        q_target = [reward[i] + discount_factor * max_q_value[i].values.numpy() for i in range(self.num_environments)]
        return q_target


    def update_delay_target_network(self):
        """
        Sets the weights of Delay-Target-Network.
        """
        self.delay_target_network.set_weights(self.network.get_weights())


    def training(self, num_training_samples, ERP):
        for _ in range(num_training_samples):
            sample_number = ERP.sample()
            q_target = self.q_target_array(ERP, sample_number)
            #print(ERP.observation[sample_number]) # tf.expand_dims funktioniert, np.shape ist (4, 84, 84), array filled with numbers
            self.network.train(ERP.observation[sample_number], q_target)
            
        self.update_delay_target_network()
    

    def training_multiple_environments(self, num_training_samples, ERPs):
        for _ in range(num_training_samples):
            sample_numbers = [ERP.sample() for ERP in ERPs]
            q_target = self.q_target_multiple_environments(ERPs, sample_numbers, discount_factor = 0.95)
            observations = tf.stack([ERPs[i].observation[sample_numbers[i]] for i in range(self.num_environments)])
            self.network.train(observations, q_target)

        # should we update this every episode?
        self.update_delay_target_network() 