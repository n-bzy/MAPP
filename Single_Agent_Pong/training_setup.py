import gymnasium as gym
import gymnasium.wrappers as gw


def hyperparameter_settings(num_actions = 6, ERP_size = 5000, EPISODES = 20_000, 
                            epsilon = 1,
                            MODEL_NAME = "SinglePong", AGGREGATE_STATS_EVERY = 1, MIN_REWARD = -21, UPDATE_TARGET_EVERY = 10):
    '''
    Parameters: 
        num_actions (int): amount of possible actions for the agent
        ERP_size (int): size of the experience replay buffer
        EPISODES (int): amount of epsiodes to train for
        For exploration:
            epsilon (float): not a constant, going to be decayed
        For logging:
            MODEL_NAME (str): used for saving and logging
            AGGREGATE_STATS_EVERY (int): get and safe stats every n episodes
            MIN_REWARD (int): safe model only when the lowest reward of model over the last n episodes reaches a threshold 

        Returns: 
            All Parameters above
    '''
    return num_actions, ERP_size, EPISODES, epsilon, MODEL_NAME, AGGREGATE_STATS_EVERY, MIN_REWARD, UPDATE_TARGET_EVERY


def create_env(name_env = 'ALE/Pong-v5'):
    '''
    Creates and preprocesses the environment to be of size (4, 84, 84, 1). 
    
    The 1st-dimension represents the 4 frames stacked together. This is for the Agent to get an idea of the direction of the ball.
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