from Experience_Replay_Buffer import ExperienceReplayBuffer
from training_setup import create_env
import numpy as np
import time

ERP_list = ExperienceReplayBuffer(5000)
ERP_array = ExperienceReplayBuffer(5000)

env1 = create_env()
env2 = create_env()

num_samples = 100

def duration_fill_up(num_samples):
    times_list = []
    times_array = []
    time_prep_list = []
    time_prep_array = []
    for _ in range(num_samples):
        times_list.append(ERP_list.fill(env1))
        start_list = time.time()
        ERP_list.preprocessing_list()
        end_list = time.time() 
        time_prep_list.append(end_list - start_list)

        times_array.append(ERP_array.fill_up(env2))
        start_array=time.time()
        ERP_array.preprocessing()
        end_array = time.time() 
        time_prep_array.append(end_array - start_array)

    print(f"average over {num_samples} ERP fills each ERP of size {ERP_list.size} with list: ", np.mean(times_list))
    print(f"average over {num_samples} ERP fills each ERP of size {ERP_array.size} with arrays: ", np.mean(times_array))

    print(f"duration preprocess list of {ERP_list.size} samples: ", np.mean(time_prep_list))
    print(f"duration preprocess array of {ERP_array.size} samples: ", np.mean(time_prep_array))

duration_fill_up(num_samples)
# average over 100 ERP fills with list:  6.2224518370628354
# average over 100 ERP fills with arrays:  7.1150505113601685

# average over 10 ERP fills with list:  6.30266227722168
# average over 10 ERP fills with arrays:  7.011609196662903

# average over 2 ERP fills each ERP of size 20000 with list:  6.154080867767334
# average over 2 ERP fills each ERP of size 20000 with arrays:  7.069062113761902

# average over 2 ERP fills each ERP of size 2000 with list:  1.0531117916107178
# average over 2 ERP fills each ERP of size 2000 with arrays:  1.1408966779708862

# average over 2 ERP fills each ERP of size 5000 with list:  1.796952247619629
# average over 2 ERP fills each ERP of size 5000 with arrays:  2.046942353248596

# average over 2 ERP fills each ERP of size 10000 with list:  3.3182393312454224
# average over 2 ERP fills each ERP of size 10000 with arrays:  3.676979660987854

# average over 2 ERP fills each ERP of size 15000 with list:  4.7899439334869385
# average over 2 ERP fills each ERP of size 15000 with arrays:  5.637444972991943


'''time_prep_list = []
time_prep_array = []

start_list = time.time()
ERP_list.preprocessing_list()
end_list = time.time() 

start_array=time.time()
ERP_array.preprocessing()
end_array = time.time() 

print(f"duration preprocess list of {ERP_list.size} samples: ", end_list - start_list)
print(f"duration preprocess array of {ERP_array.size} samples: ", end_array - start_array)'''

# 2023-03-18 08:00:33.453133: W tensorflow/tsl/framework/cpu_allocator_impl.cc:82] Allocation of 2257920000 exceeds 10% of free system memory.
# 2023-03-18 08:00:35.000815: W tensorflow/tsl/framework/cpu_allocator_impl.cc:82] Allocation of 2257920000 exceeds 10% of free system memory.
# duration preprocess list of 20000 samples:  30.700382232666016
# duration preprocess array of 20000 samples:  208.6261215209961

# duration preprocess list of 2000 samples:  2.7541956901550293
# duration preprocess array of 2000 samples:  0.33741259574890137

# duration preprocess list of 5000 samples:  6.199303865432739
# duration preprocess array of 5000 samples:  0.6063830852508545

# memory warning!!!
# duration preprocess list of 10000 samples:  13.271893501281738
# duration preprocess array of 10000 samples:  1.3080174922943115

# memory warning
# duration preprocess list of 15000 samples:  22.195309162139893
# duration preprocess array of 15000 samples:  1.8570709228515625


'''average over 100 ERP fills each ERP of size 5000 with list:  2.1212963223457337
average over 100 ERP fills each ERP of size 5000 with arrays:  2.389762659072876
duration preprocess list of 5000 samples:  1.8032095551490783
duration preprocess array of 5000 samples:  0.6083445930480957'''

'''
average over 100 ERP fills each ERP of size 5000 with list:  1.968929443359375
average over 100 ERP fills each ERP of size 5000 with arrays:  2.2281795167922973
duration preprocess list of 5000 samples:  1.6762497091293336
duration preprocess array of 5000 samples:  0.5889246320724487

'''









