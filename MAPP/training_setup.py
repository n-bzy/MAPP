from pettingzoo.atari import pong_v3
import supersuit


def hyperparameter_settings(num_actions = 6, ERP_size = 10_000, num_training_samples = 32, EPISODES = 20_000,
                            epsilon = 1, EPSILON_DECAY = 0.999985, MIN_EPSILON = 0.01,
                            MODEL_NAME = "MultiPong", UPDATE_TARGET_EVERY = 5, AGGREGATE_STATS_EVERY = 10, MIN_REWARD = 0):
    '''
    Parameters:
        num_actions (int): amount of possible actions for the agent
        ERP_size (int): size of the experience replay buffer
        num_training_samples (int): amount of samples the agent trains on every episode
        EPISODES (int): amount of epsiodes to train for
        For exploration:
            epsilon (float): not a constant, going to be decayed
            EPSILON_DECAY (float): rate to decay epsilon
            MIN_EPSILON (float): the minimum allowed for epsilon
        For logging:
            MODEL_NAME (str): used for saving and logging
            UPDATE_TARGET_EVERY (int): update the target network every n episodes
            AGGREGATE_STATS_EVERY (int): get and safe stats every n episodes
            MIN_REWARD (int): safe model only when the lowest reward of model over the last n episodes reaches a threshold
    '''
    return num_actions, ERP_size, num_training_samples,  EPISODES, epsilon, EPSILON_DECAY, MIN_EPSILON, MODEL_NAME, UPDATE_TARGET_EVERY, AGGREGATE_STATS_EVERY, MIN_REWARD


def create_env():
    '''
    Creates and preprocesses the environment to be of size (4, 84, 84, 1). 
    
    The 1st-dimension represents the 4 frames stacked together, to get an idea of the direction of the ball.
    The 2nd and 3rd dimension represent the size of the observation.
    And the 4th dimension represents the colorchannel which is grayscale.
    Return:
        env (pettingzoo): an environment to train an agent
    '''
    env = pong_v3.env(num_players=2)

    # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
    # to deal with frame flickering
    env = supersuit.max_observation_v0(env, 2)
    # repeat_action_probability is set to 0.25 to introduce non-determinism to the system
    env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)
    # Reduce color to grayscale
    env = supersuit.color_reduction_v0(env, mode='full')
    # skip frames for faster processing and less control
    env = supersuit.frame_skip_v0(env, 4)
    # downscale observation for faster processing
    env = supersuit.resize_v1(env, 84, 84)
    # Stack 4 frames in one observation
    env = supersuit.frame_stack_v1(env, 4)
    return env