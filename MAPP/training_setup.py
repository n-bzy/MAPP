import gymnasium as gym
import gymnasium.wrappers as gw


def hyperparameter_settings(num_environments = 8, num_actions = 6, ERP_size = 1000, num_training_samples = 500, TIMESTEPS = 500, EPISODES = 20_000, 
                            epsilon = 1, EPSILON_DECAY = 0.99975, MIN_EPSILON = 0.001,
                            MODEL_NAME = "SinglePong", AGGREGATE_STATS_EVERY = 50, MIN_REWARD = 0):
    '''
    Parameters: 
        num_environments (int): amount of environments
        num_actions (int): amount of possible actions for the agent
        ERP_size (int): size of the experience replay buffer
        num_training_samples (int): amount of samples the agent trains on every episode
        TIMESTEPS (int): amount of new samples to fill ERP with
        EPISODES (int): amount of epsiodes to train for
        For exploration:
            epsilon (float): not a constant, going to be decayed
            EPSILON_DECAY (float): rate to decay epsilon
            MIN_EPSILON (float): the minimum allowed for epsilon
        For logging:
            MODEL_NAME (str): used for saving and logging
            AGGREGATE_STATS_EVERY (int): get and safe stats every n episodes
            MIN_REWARD (int): safe model only when the lowest reward of model over the last n episodes reaches a threshold
    '''
    return num_environments, num_actions, ERP_size, num_training_samples, TIMESTEPS, EPISODES, epsilon, EPSILON_DECAY, MIN_EPSILON, MODEL_NAME, AGGREGATE_STATS_EVERY, MIN_REWARD 


def create_env(name_env = 'ALE/Pong-v5'):
    '''
    Creates and preprocesses the environment to be of size (4, 84, 84, 1). 
    
    The 1st-dimension represents the 4 frames stacked together, to get an idea of the direction of the ball.
    The 2nd and 3rd dimension represent the size of the observation.
    And the 4th dimension represents the colorchannel which is grayscale.
    Return:
        env (gymnasium): an environment to train an agent 
    '''
    env = gym.make(name_env) 
    env = gw.ResizeObservation(env, 84)
    env = gw.GrayScaleObservation(env, keep_dim=True)
    env = gw.FrameStack(env, 4)
    return env