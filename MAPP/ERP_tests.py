from Experience_Replay_Buffer import ExperienceReplayBuffer
from training_setup import create_env
import numpy as np

ERP_list = ExperienceReplayBuffer(20000)
ERP_array = ExperienceReplayBuffer(20000)

env1 = create_env()
env2 = create_env()

def duration_fill_up():
    times_list = []
    times_array = []
    for _ in range(100):
        times_list.append(ERP_list.fill(env1))
        times_array.append(ERP_array.fill_up(env2))

    print("average over 100 ERP fills with list: ", np.mean(times_list))
    print("average over 100 ERP fills with arrays: ", np.mean(times_array))

duration_fill_up()
# average over 100 ERP fills with list:  6.2224518370628354
# average over 100 ERP fills with arrays:  7.1150505113601685
