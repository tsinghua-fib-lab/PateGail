# encoding: utf-8

import os
import matplotlib.pyplot as plt 

import shutil
import argparse
import setproctitle
import scipy.stats
import numpy as np
from collections import Counter
from math import radians, cos, sin, asin, sqrt
from utils import get_gps, read_data_from_file, read_logs_from_file


def geodistance(lng1,lat1,lng2,lat2):
    #lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance


class EvalUtils(object):
    """
    some commonly-used evaluation tools and functions
    """

    @staticmethod
    def filter_zero(arr):
        """
        remove zero values from an array
        :param arr: np.array, input array
        :return: np.array, output array
        """
        arr = np.array(arr)
        filtered_arr = np.array(list(filter(lambda x: x != 0., arr)))
        return filtered_arr

    @staticmethod
    def arr_to_distribution(arr, min, max, bins):
        """
        convert an array to a probability distribution
        :param arr: np.array, input array
        :param min: float, minimum of converted value
        :param max: float, maximum of converted value
        :param bins: int, number of bins between min and max
        :return: np.array, output distribution array
        """
        distribution, base = np.histogram(
            arr, np.arange(
                min, max, float(
                    max - min) / bins))
        return distribution, base[:-1]

    @staticmethod
    def norm_arr_to_distribution(arr, bins=100):
        """
        normalize an array and convert it to distribution
        :param arr: np.array, input array
        :param bins: int, number of bins in [0, 1]
        :return: np.array, np.array
        """
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        distribution, base = np.histogram(arr, np.arange(0, 1, 1. / bins))
        return distribution, base[:-1]

    @staticmethod
    def log_arr_to_distribution(arr, min=-30., bins=100):
        """
        calculate the logarithmic value of an array and convert it to a distribution
        :param arr: np.array, input array
        :param bins: int, number of bins between min and max
        :return: np.array,
        """
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = EvalUtils.filter_zero(arr)
        arr = np.log(arr)
        distribution, base = np.histogram(arr, np.arange(min, 0., 1./bins))
        ret_dist, ret_base = [], []
        for i in range(bins):
            if int(distribution[i]) == 0:
                continue
            else:
                ret_dist.append(distribution[i])
                ret_base.append(base[i])
        return np.array(ret_dist), np.array(ret_base)

    @staticmethod
    def get_js_divergence(p1, p2):
        """
        calculate the Jensen-Shanon Divergence of two probability distributions
        :param p1:
        :param p2:
        :return:
        """
        # normalize
        p1 = p1 / p1.sum()
        p2 = p2 / p2.sum()
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + \
            0.5 * scipy.stats.entropy(p2, m)
        return js


class IndividualEval(object):

    def __init__(self, data):
        if data == 'mobile':       
            self.X, self.Y = get_gps('raw_data/mobile/gps')
            self.max_locs = 8606
            self.max_distance = 2.088
        elif data == 'geolife':
            self.X, self.Y = get_gps('raw_data/geolife/gps')
            self.max_locs = 4210
            self.max_distance = 6.886
   

    def get_topk_visits(self,trajs, k):
        topk_visits_loc = []
        topk_visits_freq = []
        for traj in trajs:
            topk = Counter(traj).most_common(k)
            for i in range(len(topk), k):
                # supplement with (loc=-1, freq=0)
                topk += [(-1, 0)]
            loc = [l for l, _ in topk]
            freq = [f for _, f in topk]
            loc = np.array(loc, dtype=int)
            freq = np.array(freq, dtype=float) / trajs.shape[1]
            topk_visits_loc.append(loc)
            topk_visits_freq.append(freq)
        topk_visits_loc = np.array(topk_visits_loc, dtype=int)
        topk_visits_freq = np.array(topk_visits_freq, dtype=float)
        return topk_visits_loc, topk_visits_freq

    
    def get_overall_topk_visits_freq(self, trajs, k):
        _, topk_visits_freq = self.get_topk_visits(trajs, k)
        mn = np.mean(topk_visits_freq, axis=0)
        return mn / np.sum(mn)


    def get_overall_topk_visits_loc_freq_arr(self, trajs, k=1):
        topk_visits_loc, _ = self.get_topk_visits(trajs, k)
        k_top = np.zeros(self.max_locs, dtype=float)
        for i in range(k):
            cur_k_visits = topk_visits_loc[:, i]
            for ckv in cur_k_visits:
                index = int(ckv)
                if index == -1:
                    continue
                k_top[index] += 1
        k_top = k_top / np.sum(k_top)
        return k_top

    
    def get_overall_topk_visits_loc_freq_dict(self, trajs, k):
        topk_visits_loc, _ = self.get_topk_visits(trajs, k)
        k_top = {}
        for i in range(k):
            cur_k_visits = topk_visits_loc[:, i]
            for ckv in cur_k_visits:
                index = int(ckv)
                if index in k_top:
                    k_top[int(ckv)] += 1
                else:
                    k_top[int(ckv)] = 1
        return k_top

    def get_overall_topk_visits_loc_freq_sorted(self, trajs, k, kk):
        k_top = self.get_overall_topk_visits_loc_freq_dict(trajs, k)
        k_top_list = list(k_top.items())
        k_top_list.sort(reverse=True, key=lambda k: k[1])
        return np.array(k_top_list)


    def get_geodistances(self, trajs):
        distances = []
        seq_len = 48
        for traj in trajs:
            for i in range(seq_len - 1):
                lng1 = self.X[traj[i]]
                lat1 = self.Y[traj[i]]
                lng2 = self.X[traj[i + 1]]
                lat2 = self.Y[traj[i + 1]]
                distances.append(geodistance(lng1,lat1,lng2,lat2))
        distances = np.array(distances, dtype=float)
        return distances
    
    def get_distances(self, trajs):
        distances = []
        seq_len = 48
        for traj in trajs:
            for i in range(seq_len - 1):
                dx = self.X[traj[i]] - self.X[traj[i + 1]]
                dy = self.Y[traj[i]] - self.Y[traj[i + 1]]
                distances.append(dx**2 + dy**2)
        distances = np.array(distances, dtype=float)
        return distances
    
    def get_durations(self,trajs):
        d = []
        for traj in trajs:
            num = 0
            for i in range(1,len(traj)):
                num+=1
                if traj[i]!=traj[i-1]:
                    d.append(num)
                    num = 0 
                if i == len(traj)-1:
                    d.append(num+1)
        return d
    
    def get_gradius(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :param trajs:
        :return:
        """
        gradius = []
        seq_len = 48
        for traj in trajs:
            xs = np.array([self.X[t] for t in traj])
            ys = np.array([self.Y[t] for t in traj])
            xcenter, ycenter = np.mean(xs), np.mean(ys)
            dxs = xs - xcenter
            dys = ys - ycenter
            rad = [dxs[i]**2 + dys[i]**2 for i in range(seq_len)]
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(rad)
        gradius = np.array(gradius, dtype=float)
        return gradius
    
    def get_periodicity(self, trajs):
        """
        stat how many repetitions within a single trajectory
        :param trajs:
        :return:
        """
        reps = []
        for traj in trajs:
            reps.append(float(len(set(traj)))/48)
        reps = np.array(reps, dtype=float)
        return reps

    def get_timewise_periodicity(self, trajs):
        """
        stat how many repetitions of different times
        :param trajs:
        :return:
        """
        pass


    def get_geogradius(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :param trajs:
        :return:
        """
        gradius = []
        for traj in trajs:
            xs = np.array([self.X[t] for t in traj])
            ys = np.array([self.Y[t] for t in traj])
            lng1, lat1 = np.mean(xs), np.mean(ys)
            rad = []
            for i in range(len(xs)):                   
                lng2 = xs[i]
                lat2 = ys[i]
                distance = geodistance(lng1,lat1,lng2,lat2)
                rad.append(distance)
            rad = np.mean(np.array(rad, dtype=float))
            gradius.append(rad)
        gradius = np.array(gradius, dtype=float)
        return gradius

    def get_hometime(self, trajs):
        """
        get the std of the distances of all points away from center as `gyration radius`
        :param trajs:
        :return:
        """
        hometime = []
        for traj in trajs:
            num = 0
            for t in traj:
                if self.X[t] == self.X[0]:
                    num = num+1
            hometime.append(num)
        hometime = np.array(hometime, dtype=float)
        return hometime

    def get_individual_jsds(self, t1, t2, t3):
        """
        get jsd scores of individual evaluation metrics
        :param t1:
        :param t2:
        :return:
        """
        min_distance = 0
        d1 = self.get_distances(t1)
        d2 = self.get_distances(t2)
        d3 = self.get_distances(t3)
        d1_dist, _ = EvalUtils.arr_to_distribution(
            d1, min_distance, self.max_distance, 1000)
        d2_dist, _ = EvalUtils.arr_to_distribution(
            d2, min_distance, self.max_distance, 1000)
        d3_dist, _ = EvalUtils.arr_to_distribution(
            d3, min_distance, self.max_distance, 1000)
        # print(d1_dist)
        # print(sum(d2_dist))
        #ax.set_yscale('log')
        '''
        plt.semilogy( range(len(d1_dist[0:99])), d1_dist[0:99]/sum(d1_dist),'-^', color="blue")
        plt.semilogy( range(len(d2_dist[0:99])), d2_dist[0:99]/sum(d2_dist),'-^', color="red")
        plt.semilogy( range(len(d3_dist[0:99])), d3_dist[0:99]/sum(d3_dist),'-^', color='green')
        plt.xlabel('Distance,$\Delta$r(km)')
        plt.ylabel('P($\Delta$r)')
        plt.savefig('distance.png')
        '''
        d_jsd = EvalUtils.get_js_divergence(d1_dist, d2_dist)
        
        
        
        p1 = self.get_periodicity(t1)
        p2 = self.get_periodicity(t2)
        p1_dist, _ = EvalUtils.arr_to_distribution(p1, 0, 1, 48)
        p2_dist, _ = EvalUtils.arr_to_distribution(p2, 0, 1, 48)
        p_jsd = EvalUtils.get_js_divergence(p1_dist, p2_dist)
        p3 = self.get_periodicity(t3)
        p3_dist, _ = EvalUtils.arr_to_distribution(p3, 0, 1, 48)
        p11=[]
        p12=[]
        p13=[]
        '''
        for i in range(len(p1_dist)):
            p11.append(sum((p1_dist[0:i])))
            p12.append(sum((p2_dist[0:i])))
            p13.append(sum((p3_dist[0:i])))
        plt.plot( range(len(p1_dist)), p11/sum(p1_dist),'-^', color="blue")
        plt.plot( range(len(p2_dist)), p12/sum(p2_dist),'-^', color="red")
        plt.plot(range(len(p3_dist)), p13/sum(p3_dist), '-^', color='green')
        plt.xlabel('Locations,N')
        plt.ylabel('CDF')
        plt.savefig('dailyloc.png')        
        '''
        du1 = self.get_durations(t1)
        du2 = self.get_durations(t2)     
        du1_dist, _ = EvalUtils.arr_to_distribution(du1, 1, 49, 48)
        du2_dist, _ = EvalUtils.arr_to_distribution(du2, 1, 49, 48)
        du_jsd = EvalUtils.get_js_divergence(du1_dist, du2_dist)
        du3 = self.get_durations(t3)     
        '''
        plt.semilogy( range(len(d1_dist[0:23])), d1_dist[0:23]/sum(d1_dist[0:23]),'-^', color="blue")
        plt.semilogy( range(len(d2_dist[0:23])), d2_dist[0:23]/sum(d2_dist[0:23]),'-^', color="red")
        plt.semilogy( range(len(d3_dist[0:23])), d3_dist[0:23]/sum(d3_dist[0:23]),'-^', color='green')
        plt.xlabel('Duration,$\Delta$t(hour)')
        plt.ylabel('P($\Delta$t)')
        plt.savefig('duration.png')
        '''
        
        a1 = self.get_hometime(t1)
        a2 = self.get_hometime(t2)
        a1_dist, _ = EvalUtils.arr_to_distribution(a1, 1, 49, 48)
        a2_dist, _ = EvalUtils.arr_to_distribution(a2, 1, 49, 48)
        a_jsd = EvalUtils.get_js_divergence(a1_dist, a2_dist)        
        
        
        
        g1 = self.get_gradius(t1)
        g2 = self.get_gradius(t2)
        g1_dist, _ = EvalUtils.arr_to_distribution(
            g1, min_distance, self.max_distance, 1000)
        g2_dist, _ = EvalUtils.arr_to_distribution(
            g2, min_distance, self.max_distance, 1000)
        g3 = self.get_gradius(t3)
        g3_dist, _ = EvalUtils.arr_to_distribution(
            g3, min_distance, self.max_distance, 1000)        
        g_jsd = EvalUtils.get_js_divergence(g1_dist, g2_dist)


        # print(sum(g1_dist),sum(g2_dist),sum(g3_dist))
        '''
        plt.semilogy( range(len(g1_dist[0:29])), g1_dist[0:29]/sum(g1_dist),'-^', color="blue")
        plt.semilogy( range(len(g2_dist[0:29])), g2_dist[0:29]/sum(g2_dist),'-^', color="red")
        plt.semilogy(range(len(g3_dist[0:29])), g3_dist[0:29]/sum(g3_dist), '-^', color='green')
        plt.xlabel('Radius,$\Delta$r(km)')
        plt.ylabel('P($\Delta$r)')
        plt.savefig('radius.png')
        '''
        l1 =  CollectiveEval.get_visits(t1,self.max_locs)
        l2 =  CollectiveEval.get_visits(t2,self.max_locs)
        l3 =  CollectiveEval.get_visits(t3,self.max_locs)
        l1_dist = CollectiveEval.get_topk_visits(l1, 50)
        l2_dist = CollectiveEval.get_topk_visits(l2, 50)
        l3_dist = CollectiveEval.get_topk_visits(l3, 50)
        l1_dist,_ = EvalUtils.arr_to_distribution(l1_dist,0,1,50)
        l2_dist,_ = EvalUtils.arr_to_distribution(l2_dist,0,1,50)
        l3_dist,_ = EvalUtils.arr_to_distribution(l3_dist,0,1,50)
        '''
        plt.plot( range(len(l1_dist[0:19])), l1_dist[0:19]/sum(l1_dist[0:19]),'-^', color="blue")
        plt.plot( range(len(l2_dist[0:19])), l2_dist[0:19]/sum(l2_dist[0:19]),'-^', color="red")
        plt.plot(range(len(l3_dist[0:19])), l3_dist[0:19]/sum(l3_dist[0:19]), '-^', color='green')
        plt.xlabel('G-rank,locations')
        plt.ylabel('P')
        plt.savefig('Grank.png')
        '''
        l_jsd = EvalUtils.get_js_divergence(l1_dist, l2_dist)

        f1 = self.get_overall_topk_visits_freq(t1, 20)
        f2 = self.get_overall_topk_visits_freq(t2, 20)
        f3 = self.get_overall_topk_visits_freq(t3, 20)
        f_jsd = EvalUtils.get_js_divergence(f1, f2)
        np.save('d1_dist.npy',f1)
        np.save('d2_dist.npy',f2)
        du3_dist, _ = EvalUtils.arr_to_distribution(du1, 1, 49, 48)
        np.save('d3_dist.npy',f3)
        '''
        plt.plot( range(len(f1[0:7])), f1[0:7]/sum(f1[0:7]),'-^', color="blue")
        plt.plot( range(len(f2[0:7])), f2[0:7]/sum(f2[0:7]),'-^', color="red")
        plt.plot(range(len(f3[0:7])), f3[0:7]/sum(f3[0:7]), '-^', color='green')
        plt.xlabel('I-rank,rank')
        plt.ylabel('P')
        plt.savefig('Irank.png')
        '''
        print('distance:',d_jsd,'radius:', g_jsd,'daily-loc:',p_jsd,'duration:', du_jsd ,'irank:', f_jsd,'grank:',l_jsd)
        return d_jsd, p_jsd, g_jsd, l_jsd, f_jsd, du_jsd, a_jsd




class CollectiveEval(object):
    """
    collective evaluation metrics
    """
    @staticmethod
    def get_visits(trajs,max_locs):
        """
        get probability distribution of visiting all locations
        :param trajs:
        :return:
        """
        visits = np.zeros(shape=(max_locs), dtype=float)
        for traj in trajs:
            for t in traj:
                visits[t] += 1
        visits = visits / np.sum(visits)
        return visits

    @staticmethod
    def get_timewise_visits(trajs):
        """
        stat how many visits of a certain location in a certain time
        :param trajs:
        :return:
        """
        pass

    @staticmethod
    def get_topk_visits(visits, K):
        """
        get top-k visits and the corresponding locations
        :param trajs:
        :param K:
        :return:
        """
        locs_visits = [[i, visits[i]] for i in range(visits.shape[0])]
        locs_visits.sort(reverse=True, key=lambda d: d[1])
        topk_locs = [locs_visits[i][0] for i in range(K)]
        topk_probs = [locs_visits[i][1] for i in range(K)]
        return np.array(topk_probs), topk_locs

    @staticmethod
    def get_topk_accuracy(v1, v2, K):
        """
        get the accuracy of top-k visiting locations
        :param v1:
        :param v2:
        :param K:
        :return:
        """
        _, tl1 = CollectiveEval.get_topk_visits(v1, K)
        _, tl2 = CollectiveEval.get_topk_visits(v2, K)
        coml = set(tl1) & set(tl2)
        return len(coml) / K


def evaluate(datasets, gene_data):
    if datasets == 'telecom':
        individualEval = IndividualEval(data='mobile')
        start_point = np.load('../data/mobile/start.npy')
    else:
        individualEval = IndividualEval(data='geolife')
        # start_point = np.load('../data/geolife/start.npy')    
    
    test_data = read_data_from_file('raw_data/%s/test.data' % opt.datasets)
    
    base = read_data_from_file('raw_data/%s/test.data' % opt.datasets)
    #print(test_data[1])
    #print(gene_data[1])
    test_data = test_data
    print(test_data.shape)
    print(individualEval.get_individual_jsds(test_data,gene_data, base))
    

if __name__ == "__main__":
    # global
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',default='default', type=str)
    parser.add_argument('--cuda',default=0,type=int)
    parser.add_argument('--datasets',default='geolife',type=str)
    opt = parser.parse_args()
    if opt.datasets == 'geolife':
        max_locs = 4210
    else:
        max_locs = 8606
    print(max_locs)
    # gene_data = np.load('./results/eval_3.1_50/eval_1.npy')
    # print(gene_data.shape)
    # evaluate(opt.datasets, gene_data)
    import glob
    files = glob.glob('./eval*/')
    for file in files:
        dd = glob.glob(file+'*.npy')
        if dd:
            for ff in dd:
                gene_data = np.load(ff)
                evaluate(opt.datasets, gene_data)
            print(f'file_from:{dd}', '----------------------------------')
    #evaluate(opt.datasets, gene_data)
