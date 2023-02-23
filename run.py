from models.mp_gail import gail
from env.env import timegeo_env
import numpy as np
import argparse
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='parameters of gail')
    parser.add_argument('--seed',type=int, default= 9999)
    parser.add_argument('--beta', type = float, default= 0.01)
    parser.add_argument('--noise', type = float, default= 0.01)
    args = parser.parse_args()
    env = timegeo_env()
    file = np.load('./dataset/geolife/real_data.npy',allow_pickle=True).item()
    eval = False
    test = gail(
        env=env,
        file=file,
        eval = eval,
        seed = args.seed,
        beta = args.beta,
        noise = args.noise        
    )
    if eval:
        test.eval_data()
    else:
        test.run()