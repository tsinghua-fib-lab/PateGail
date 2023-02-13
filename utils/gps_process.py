import numpy as np
from geopy.distance import geodesic
from multiprocessing import Process
import os


def calc_rank_matrix(gps_info, start_idx, end_idx):
    distance_matrix = []
    for i in range(start_idx, end_idx):
        distance_matrix.append([])
        for j in range(len(gps_info)):
            distance_matrix[-1].append(geodesic(gps_info[i], gps_info[j]).m)
        if start_idx == 0:
            print('finish {} / {}'.format(i, end_idx))
    np.save('{}_{}'.format(start_idx, end_idx), distance_matrix)


if __name__ == '__main__':
    data = 'geolife'
    os.makedirs('../preprocess_data/{}'.format(data), exist_ok=True)
    gps_info = np.loadtxt('../raw_data/{}/gps'.format(data))
    track_data = np.loadtxt('../raw_data/{}/real.data'.format(data))
    process_num = 55
    index_list = np.linspace(0, len(gps_info), process_num).astype(int)
    for i in range(len(gps_info)):
        gps_info[i, 0] = gps_info[i, 0] - 90
    print(gps_info)
    process_list = []
    for index in range(process_num - 1):
        p = Process(target=calc_rank_matrix, args=(gps_info, index_list[index], index_list[index + 1]))
        process_list.append(p)

    for i, p in enumerate(process_list):
        p.start()
        print('{} start'.format(i))
    for i, p in enumerate(process_list):
        p.join()
        print('{} join'.format(i))
    

    distance_matrix = np.zeros([len(gps_info), len(gps_info)])
    for i in range(process_num - 1):
        distance_matrix[index_list[i]: index_list[i + 1]] = np.array(np.load('{}_{}.npy'.format(index_list[i], index_list[i + 1])))

    np.save('../preprocess_data/{}/distance_matrix'.format(data), distance_matrix)

    rank_list = []
    for i in range(len(distance_matrix)):
        rank_list.append(list(distance_matrix[i].argsort()))
        print('{} / {}'.format(i, len(distance_matrix)))

    np.save('../preprocess_data/{}/rank_list'.format(data), rank_list)
