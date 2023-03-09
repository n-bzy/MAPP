import numpy as np

class ExperienceReplayBuffer():
    def __init__(self, size): 

        self.size = size 
        self.experience_replay_buffer = [0] * self.size # maybe an array is nice
        self.index = 0

    def sample(self):
        """
        Randomly choose a sample from the Experience-Replay-Buffer.
        """
        sample = self.experience_replay_buffer[np.random.randint(self.size)] 
        return sample

    







