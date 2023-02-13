import numpy as np
from itertools import groupby
from multiprocessing import Process
import os


if __name__ == '__main__':
    data = 'geolife'
    os.makedirs('../preprocess_data/geolife/'.format(data), exist_ok=True)
    track_data = np.loadtxt('../raw_data/geolife/real.data'.format(data)).astype(int)
    rank_list = np.load('../preprocess_data/geolife/rank_list.npy'.format(data))
    gps_info = np.loadtxt('../raw_data/geolife/gps'.format(data))

    # * all burst list
    burst_list = []
    for i in range(len(track_data)):
        burst_list.append([x[0] for x in groupby(track_data[i])])
    np.save('../preprocess_data/geolife/burst_list'.format(data), burst_list)

    # * all burst rank list
    burst_rank_list = []
    for i in range(len(burst_list)):
        for j in range(len(burst_list[i]) - 1):
            start_pos = burst_list[i][j]
            end_pos = burst_list[i][j + 1]
            burst_rank_list.append(list(rank_list[start_pos]).index(end_pos))
        print('{} / {}'.format(i, len(burst_list)))
    np.save('../preprocess_data/geolife/burst_rank_list'.format(data), burst_rank_list)

    # * burst matrix
    burst_matrix = np.zeros([len(gps_info), len(gps_info)])
    for i in range(len(track_data)):
        for j in range(track_data.shape[1] - 1):
            burst_matrix[track_data[i, j], track_data[i, j + 1]] += 1
        print('finish {} / {}'.format(i, len(track_data)))
    np.save('../preprocess_data/geolife/burst_matrix'.format(data), burst_matrix)
