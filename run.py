from models.mp_gail import gail
from env.env import timegeo_env
import numpy as np
if __name__ == '__main__':
    #mp.set_start_method('spawn')
    env = timegeo_env()
    #file = np.loadtxt('./raw_data/geolife/real.data')
    file = np.load('./real_data.npy',allow_pickle=True).item()
    test = gail(
        env=env,
        file=file
    )
    test.run()
